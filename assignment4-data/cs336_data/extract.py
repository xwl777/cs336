"""把 HTML 字节解码成字符串，再提取可见文本。"""

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """优先 UTF-8；失败时检测编码。返回提取文本，不包含归档或 HTTP 头。"""
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # 优先参考 HTML 的 charset 声明，再由检测器判断编码。
        encoding = detect_encoding(html_bytes, from_html_meta=True)
        if not encoding:
            encoding = "utf-8"
        # 损坏字节显示为 �，避免整张网页因个别坏字节而丢失。
        html = html_bytes.decode(encoding, errors="replace")

    text = extract_plain_text(html, main_content=False, links=False)
    return text
