import pytest

from .adapters import run_extract_text_from_html_bytes


@pytest.mark.parametrize(
    "encoding, text",
    [
        ("utf-8", "中文与 café 都应保留。"),
        ("windows-1252", "Un café coûte cinq euros. Voilà le résumé."),
        ("gb18030", "网页正文应当正确解码，不应该出现乱码。"),
        ("shift_jis", "これは日本語の本文です。文字化けしてはいけません。"),
    ],
)
def test_extract_text_from_html_bytes_encodings(encoding, text):
    html = f'<html><head><meta charset="{encoding}"></head><body><main><p>{text}</p></main></body></html>'
    extracted = run_extract_text_from_html_bytes(html.encode(encoding))
    assert extracted == text


def test_extract_text_from_html_bytes_preserves_visible_text():
    html = b"""<html><body>
    <nav><a href='/login'>LOGIN</a><a href='/signup'>SIGNUP</a></nav>
    <main><h1>Article</h1><p>A complete paragraph about how flowers grow in spring.</p></main>
    <script>console.log('SCRIPT_TEXT');</script>
    <footer>FOOTER_TEXT</footer></body></html>"""
    extracted = run_extract_text_from_html_bytes(html)
    assert "A complete paragraph about how flowers grow in spring." in extracted
    assert "LOGIN" in extracted
    assert "FOOTER_TEXT" in extracted
    assert "SCRIPT_TEXT" not in extracted
    assert "/login" not in extracted


def test_extract_text_from_html_bytes_empty():
    assert run_extract_text_from_html_bytes(b"") == ""
