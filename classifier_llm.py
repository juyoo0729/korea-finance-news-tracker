VALID_SECTORS = [
    "반도체", "자동차", "이차전지", "바이오/제약", "금융",
    "게임/엔터", "조선/방산", "철강/소재", "부동산/건설", "IT/플랫폼", "기타",
]

_PROMPT = """\
다음 한국 경제 뉴스 헤드라인의 산업 섹터를 분류하세요.
선택지: [반도체, 자동차, 이차전지, 바이오/제약, 금융, 게임/엔터, 조선/방산, 철강/소재, 부동산/건설, IT/플랫폼, 기타]
헤드라인: {text}
답변은 위 선택지 중 하나만 정확히. 부연 설명 금지."""


def classify_by_llm(text: str, model: str = "exaone3.5:2.4b") -> str:
    try:
        import ollama
        response = ollama.generate(model=model, prompt=_PROMPT.format(text=text))
        result = response["response"].strip()
        for sector in VALID_SECTORS:
            if sector in result:
                return sector
        return "기타"
    except Exception:
        return "기타"
