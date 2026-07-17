"""외부 문자열을 Streamlit HTML에 안전하게 삽입하기 위한 도우미."""

from html import escape
from urllib.parse import quote, urlparse


def escape_html(value: object) -> str:
    """텍스트와 HTML 속성에 사용할 수 있도록 특수문자를 이스케이프한다."""
    return escape(str(value), quote=True)


def safe_http_url(value: object) -> str:
    """절대 HTTP(S) URL만 허용하고 HTML 속성용 문자열로 반환한다."""
    raw = str(value or "").strip()
    try:
        parsed = urlparse(raw)
    except ValueError:
        return "#"
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return "#"
    encoded = quote(raw, safe=":/?#[]@!$&'()*+,;=%")
    return escape_html(encoded)
