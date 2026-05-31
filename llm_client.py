import os

_GPT_DEFAULT = "gpt-5.4-mini"


def _generate(prompt: str, model: str | None = None) -> str:
    return _call_openai(prompt, model or _GPT_DEFAULT)


def _call_openai(prompt: str, model: str) -> str:
    import openai
    api_key = os.environ["OPENAI_API_KEY"]
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
