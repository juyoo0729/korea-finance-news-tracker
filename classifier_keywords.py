import yaml
from pathlib import Path
from typing import Optional

_SECTORS: dict | None = None


def _load_sectors() -> dict:
    global _SECTORS
    if _SECTORS is None:
        yaml_path = Path(__file__).parent / "sectors.yaml"
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _SECTORS = data["sectors"]
    return _SECTORS


def classify_by_keywords(text: str) -> Optional[str]:
    sectors = _load_sectors()
    text_lower = text.lower()

    scores: dict[str, int] = {}
    for sector, keywords in sectors.items():
        count = sum(1 for kw in keywords if kw.lower() in text_lower)
        if count > 0:
            scores[sector] = count

    if not scores:
        return None

    return max(scores, key=lambda s: scores[s])
