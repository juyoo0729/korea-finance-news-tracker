import unittest

from web_safety import escape_html, safe_http_url


class WebSafetyTests(unittest.TestCase):
    def test_escape_html_quotes_and_tags(self):
        self.assertEqual(
            escape_html('<img src=x onerror="alert(1)">'),
            "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;",
        )

    def test_safe_http_url_rejects_script_scheme(self):
        self.assertEqual(safe_http_url("javascript:alert(1)"), "#")

    def test_safe_http_url_allows_and_attribute_escapes_https(self):
        self.assertEqual(
            safe_http_url('https://example.com/news?a=1&b="x"'),
            "https://example.com/news?a=1&amp;b=%22x%22",
        )


if __name__ == "__main__":
    unittest.main()
