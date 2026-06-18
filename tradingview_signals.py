"""
TradingView webhook signal storage and lookup.

This module keeps TradingView separate from the app's own scoring and chart
logic. It stores incoming webhook payloads in SQLite and exposes only the latest
valid signal for a selected ticker.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


DB_PATH = Path(__file__).parent / "data_cache" / "tradingview_signals.sqlite3"
WEBHOOK_PATH = "/tradingview-webhook"
DEFAULT_WEBHOOK_PORT = 8787
DEFAULT_VALID_MINUTES = 1440
KST = ZoneInfo("Asia/Seoul")

SIGNAL_MAP = {
    "BUY": "BUY",
    "SELL": "SELL",
    "NEUTRAL": "NEUTRAL",
    "UP": "BUY",
    "DOWN": "SELL",
}

REASON_LABELS = {
    "MA5_MA20_GOLDEN_CROSS": "MA5가 MA20을 상향 돌파",
    "MA5_MA20_DEAD_CROSS": "MA5가 MA20을 하향 돌파",
    "RSI_30_CROSS_UP": "RSI가 30을 상향 돌파",
    "RSI_70_CROSS_DOWN": "RSI가 70을 하향 돌파",
}

SIGNAL_LABELS = {
    "BUY": ("🟢 기술적 매수 신호", "#00D9A3"),
    "SELL": ("🔴 기술적 매도 신호", "#FF4B5C"),
    "NEUTRAL": ("⚪ 중립", "#8B92A6"),
}

_server: ThreadingHTTPServer | None = None
_server_lock = threading.Lock()


@dataclass(frozen=True)
class WebhookServerStatus:
    running: bool
    url: str
    message: str


def normalize_symbol(symbol: str | None) -> str:
    """Normalize KRX:005930, 005930.KS, 005930.KQ, and 005930 to 005930."""
    value = str(symbol or "").strip().upper()
    if ":" in value:
        value = value.split(":", 1)[1]
    for suffix in (".KS", ".KQ"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits.zfill(6) if digits else value


def normalize_signal(signal: str | None) -> str:
    value = str(signal or "").strip().upper()
    if value not in SIGNAL_MAP:
        raise ValueError(f"unsupported signal: {signal}")
    return SIGNAL_MAP[value]


def reason_to_korean(reason: str | None) -> str:
    value = str(reason or "").strip()
    return REASON_LABELS.get(value, value or "조건 정보 없음")


def signal_label(signal: str) -> tuple[str, str]:
    return SIGNAL_LABELS.get(signal, ("⚪ 중립", "#8B92A6"))


def get_valid_minutes() -> int:
    try:
        return max(1, int(os.getenv("TRADINGVIEW_SIGNAL_VALID_MINUTES", DEFAULT_VALID_MINUTES)))
    except ValueError:
        return DEFAULT_VALID_MINUTES


def _parse_event_time(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)

    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    return parsed.astimezone(timezone.utc)


def format_event_time(value: str | None) -> str:
    if not value:
        return "-"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(KST).strftime("%Y-%m-%d %H:%M")


def format_price(value: Any) -> str:
    try:
        return f"{float(value):,.0f}원"
    except (TypeError, ValueError):
        return "-"


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tradingview_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                symbol_raw TEXT NOT NULL,
                symbol_normalized TEXT NOT NULL,
                name TEXT,
                timeframe TEXT,
                signal_raw TEXT NOT NULL,
                signal_normalized TEXT NOT NULL,
                price REAL,
                event_time_utc TEXT NOT NULL,
                reason TEXT,
                payload_json TEXT NOT NULL,
                received_at_utc TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tradingview_signals_symbol_time
            ON tradingview_signals (symbol_normalized, event_time_utc DESC)
            """
        )


def save_webhook_payload(payload: dict[str, Any]) -> int:
    init_db()
    symbol_raw = str(payload.get("symbol") or "").strip()
    symbol_normalized = normalize_symbol(symbol_raw)
    if not symbol_normalized:
        raise ValueError("symbol is required")

    signal_raw = str(payload.get("signal") or "").strip().upper()
    signal_normalized = normalize_signal(signal_raw)
    event_time = _parse_event_time(payload.get("event_time"))

    price_value = payload.get("price")
    try:
        price = float(price_value) if price_value not in (None, "") else None
    except (TypeError, ValueError):
        price = None

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO tradingview_signals (
                source, symbol_raw, symbol_normalized, name, timeframe,
                signal_raw, signal_normalized, price, event_time_utc,
                reason, payload_json, received_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(payload.get("source") or "tradingview"),
                symbol_raw,
                symbol_normalized,
                str(payload.get("name") or ""),
                str(payload.get("timeframe") or ""),
                signal_raw,
                signal_normalized,
                price,
                event_time.isoformat(),
                str(payload.get("reason") or ""),
                json.dumps(payload, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return int(cur.lastrowid)


def get_latest_signal_for_symbol(symbol: str) -> dict[str, Any] | None:
    init_db()
    normalized = normalize_symbol(symbol)
    if not normalized:
        return None

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT *
            FROM tradingview_signals
            WHERE symbol_normalized = ?
            ORDER BY event_time_utc DESC, id DESC
            LIMIT 1
            """,
            (normalized,),
        ).fetchone()

    if row is None:
        return None

    item = dict(row)
    event_time = datetime.fromisoformat(item["event_time_utc"])
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    item["is_expired"] = datetime.now(timezone.utc) - event_time > timedelta(
        minutes=get_valid_minutes()
    )
    return item


def _webhook_secret() -> str:
    return os.getenv("TRADINGVIEW_WEBHOOK_SECRET", "webhook-secret")


def _is_secret_valid(payload: dict[str, Any]) -> bool:
    expected = _webhook_secret()
    return str(payload.get("secret") or "") == expected


class _TradingViewWebhookHandler(BaseHTTPRequestHandler):
    server_version = "TradingViewWebhook/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == WEBHOOK_PATH:
            self._send_json(200, {"ok": True, "message": "TradingView webhook endpoint is running"})
        else:
            self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != WEBHOOK_PATH:
            self._send_json(404, {"ok": False, "error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload must be a JSON object")
            if not _is_secret_valid(payload):
                self._send_json(401, {"ok": False, "error": "invalid secret"})
                return
            row_id = save_webhook_payload(payload)
            self._send_json(200, {"ok": True, "id": row_id})
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})


def start_webhook_server() -> WebhookServerStatus:
    global _server
    port = int(os.getenv("TRADINGVIEW_WEBHOOK_PORT", str(DEFAULT_WEBHOOK_PORT)))
    url = f"http://localhost:{port}{WEBHOOK_PATH}"

    with _server_lock:
        init_db()
        if _server is not None:
            return WebhookServerStatus(True, url, "running")
        try:
            _server = ThreadingHTTPServer(("localhost", port), _TradingViewWebhookHandler)
        except OSError as exc:
            return WebhookServerStatus(False, url, str(exc))

        thread = threading.Thread(target=_server.serve_forever, daemon=True)
        thread.start()
        return WebhookServerStatus(True, url, "started")
