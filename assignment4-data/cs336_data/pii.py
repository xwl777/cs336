"""用可读的正则屏蔽常见邮箱、美国电话号码和 IPv4；不保证覆盖所有 PII。"""

import re


# 常见未加引号的 ASCII 邮箱：用户名的点分段 + 至少含一个点的域名。
_EMAIL_ATOM = r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
_DOMAIN_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
_EMAIL_PATTERN = re.compile(
    rf"(?<![\w.+-])"
    rf"{_EMAIL_ATOM}(?:\.{_EMAIL_ATOM})*"
    rf"@(?:{_DOMAIN_LABEL}\.)+[A-Za-z]{{2,63}}"
    rf"(?![\w-])"
)

# 支持 10 位本地号码、可选美国国家码1、括号/空格/横线/点、分机。
# 不把严格的号码分配规则当检测条件：题目也包含未必真实分配的号码。
_PHONE_PATTERN = re.compile(
    r"""
    (?<![\w+])
    (?: \+?1 [ \t.-]* )?                  # 可选国家码 1 或 +1
    (?:
        \([2-9][0-9]{2}\) [ \t.-]*        # 带括号的区号
        | [2-9][0-9]{2} [ \t.-]*          # 不带括号的区号
    )
    [0-9]{3} [ \t.-]* [0-9]{4}            # 后7位
    (?: [ \t]* (?:ext\.?|extension|x) [ \t]* [0-9]{1,6} )?
    (?!\w)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# 每段限制为0–255，不允许把非法地址的一部分当合法地址匹配。
# 本规则使用常见十进制写法，不接受 001 之类的前导零写法。
_IP_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])"
_IP_PATTERN = re.compile(
    rf"(?<![\w.]){_IP_OCTET}(?:\.{_IP_OCTET}){{3}}(?!\w|\.[0-9])"
)


def mask_emails(text: str) -> tuple[str, int]:
    # subn 一次返回替换后的字符串和本次实际替换次数。
    masked_text, count = _EMAIL_PATTERN.subn("|||EMAIL_ADDRESS|||", text)
    return masked_text, count


def mask_phone_numbers(text: str) -> tuple[str, int]:
    masked_text, count = _PHONE_PATTERN.subn("|||PHONE_NUMBER|||", text)
    return masked_text, count


def mask_ips(text: str) -> tuple[str, int]:
    masked_text, count = _IP_PATTERN.subn("|||IP_ADDRESS|||", text)
    return masked_text, count
