"""
news_tracker.py — RSS 뉴스 수집 + 키워드 필터링 + Ollama 요약 → Markdown 저장

사용법:
  python news_tracker.py                        # 전체 기사 저장 (요약 없음)
  python news_tracker.py --keyword 삼성전자     # 키워드 필터링
  python news_tracker.py --keyword 삼성전자 --summarize
  python news_tracker.py --keyword SK하이닉스 --summarize --model llama3.2:3b
  python news_tracker.py --output-dir ./reports
"""

import sys
import argparse
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Windows에서 이모지·한글 깨짐 방지: stdout을 UTF-8로 교체 후 Console에 주입
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    _stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stdout = _stdout
else:
    _stdout = sys.stdout

console = Console(file=_stdout, force_terminal=True, highlight=False)

from hankyung_feed import fetch_hankyung_finance

SUMMARY_PROMPT = """\
[지시사항]
- 출력 언어: 한국어만 사용. 영어, 한자, 외래어 사용 금지.
- 형식: 세 줄. 각 줄은 완결된 문장 하나. 번호·기호 없음.
- 근거: 아래 헤드라인에 있는 내용만 사용. 없는 내용 추가 금지.

[헤드라인]
{headlines}

[요청]
위 헤드라인에서 '{keyword}' 관련 핵심 흐름을 한국어로 세 줄 요약하라."""


def summarize(headlines: list[str], keyword: str, model: str) -> str:
    try:
        import ollama
    except ImportError:
        console.print("[bold red]오류:[/] ollama 패키지가 없습니다. 'pip install ollama' 실행 후 재시도하세요.")
        raise SystemExit(1)

    prompt = SUMMARY_PROMPT.format(
        headlines="\n".join(f"- {h}" for h in headlines),
        keyword=keyword,
    )
    try:
        response = ollama.generate(model=model, prompt=prompt)
    except Exception as e:
        console.print(f"[bold red]Ollama 오류:[/] {e}")
        console.print("'ollama serve' 가 실행 중인지, 모델이 설치됐는지 확인하세요.")
        raise SystemExit(1)
    return response["response"].strip()


def filter_articles(articles: list[dict], keyword: str) -> list[dict]:
    kw = keyword.lower()
    return [a for a in articles if kw in a["title"].lower()]


def save_to_markdown(
    keyword: str | None,
    articles: list[dict],
    output_dir: Path,
    summary: str | None = None,
) -> Path:
    today = date.today().isoformat()
    path = output_dir / f"news_{today}.md"

    header = f"# 한경 금융 뉴스 — {today}"
    if keyword:
        header += f" / 키워드: {keyword}"

    summary_block = f"## 요약\n\n{summary}\n\n---\n\n" if summary else ""

    headlines_md = "\n".join(
        f"- [{a['title']}]({a['link']})  \n  _{a['published']}_"
        for a in articles
    )

    new_block = f"{header}\n\n{summary_block}{headlines_md}\n\n---\n\n"

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(new_block + existing, encoding="utf-8")
    return path


def print_step(step: int, total: int, message: str) -> None:
    console.print(f"[bold cyan]\\[{step}/{total}][/] {message}")


def print_stats(total: int, keyword: str | None, filtered: int | None) -> None:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    table.add_column("항목", style="dim")
    table.add_column("건수", justify="right", style="bold green")

    table.add_row("수집 기사", str(total))
    if keyword and filtered is not None:
        table.add_row(f'"{keyword}" 필터 후', str(filtered))

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="한경 금융 뉴스 수집 → Markdown 저장")
    parser.add_argument("--keyword",    default=None,             help="필터링할 키워드 (없으면 전체 저장)")
    parser.add_argument("--summarize",  action="store_true",      help="Ollama로 3줄 요약 생성")
    parser.add_argument("--model",      default="exaone3.5:2.4b", help="Ollama 모델명")
    parser.add_argument("--max-items",  type=int, default=50,     help="수집할 최대 기사 수")
    parser.add_argument("--output-dir", default=".",              help="Markdown 저장 폴더")
    args = parser.parse_args()

    total_steps = 2 + bool(args.summarize)

    # 1) 수집
    print_step(1, total_steps, f"뉴스 수집 중... (최대 [yellow]{args.max_items}[/]건)")
    articles = fetch_hankyung_finance(max_items=args.max_items)

    filtered_count = None
    if args.keyword:
        articles = filter_articles(articles, args.keyword)
        filtered_count = len(articles)

    print_stats(args.max_items, args.keyword, filtered_count)

    # 2) 요약 (선택)
    summary = None
    if args.summarize:
        kw = args.keyword or "전체"
        if not articles:
            console.print("[yellow]기사가 없어 요약을 건너뜁니다.[/]")
        else:
            print_step(2, total_steps, f"[bold]'{kw}'[/] 요약 생성 중 ([dim]{args.model}[/])...")
            summary = summarize([a["title"] for a in articles], kw, args.model)
            console.print(
                Panel(
                    summary,
                    title="[bold yellow]📰 오늘의 핵심 흐름[/]",
                    border_style="yellow",
                    padding=(1, 2),
                )
            )

    # 3) 저장
    print_step(total_steps, total_steps, "Markdown 저장 중...")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = save_to_markdown(args.keyword, articles, out_dir, summary)
    console.print(f"      [bold green]저장 완료:[/] {md_path}")


if __name__ == "__main__":
    main()
