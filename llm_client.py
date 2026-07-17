import os

# Claude(Anthropic)로 전환. 요약·분류용이라 빠르고 저렴한 Haiku 4.5 기본.
_CLAUDE_DEFAULT = "claude-haiku-4-5-20251001"


def _generate(prompt: str, model: str | None = None) -> str:
    return _call_claude(prompt, model or _CLAUDE_DEFAULT)


def _call_claude(prompt: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        b.text for b in resp.content if getattr(b, "type", "") == "text"
    ).strip()
