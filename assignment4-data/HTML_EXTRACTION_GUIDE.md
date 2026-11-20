# HTML 字节怎样变成文本：对照 20 行代码看

实现：[cs336_data/extract.py](cs336_data/extract.py)。adapter：[tests/adapters.py](tests/adapters.py) 中的 `run_extract_text_from_html_bytes`。整文件比较脚本：[scripts/compare_html_extraction.py](scripts/compare_html_extraction.py)。

## 1. 先分清两个任务：认出文字，再从 HTML 取文字

下载到的 HTML 起初是 bytes，也就是一串字节，不是 Python 已经认得的文字。它可能表示 UTF-8 中文，也可能表示 Windows-1252 法文、GB18030 中文或 Shift-JIS 日文。

所以先要解决“这些字节应该按哪种编码读”，得到 HTML 字符串；然后才解决“哪些内容是标签、哪些是可见文字”。

```text
原始 bytes
  → 按正确编码 decode
  → HTML 字符串，例如 <p>你好</p>
  → Resiliparse 提取
  → 文本字符串，例如 你好
```

解码和提取是两件事。UTF-8 能正确解码，不代表文字已经没有 HTML 标签；标签删掉了，也不代表导航、页脚等杂质已经被清掉。

## 2. 第 3、4 行：两个库函数分别干什么

```python
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
```

`detect_encoding` 帮我们猜字节编码。`extract_plain_text` 解析 HTML 并提取文字。不用自己写正则删除标签，也不用自己实现 HTML 解析器。

## 3. 第 7 行：输入输出是什么

```python
def extract_text_from_html_bytes(html_bytes: bytes) -> str:
```

`html_bytes` 应该是 **HTTP 响应中的 HTML 正文**。不是文件名，不是整个压缩文件，也不是包含 WARC 头和 HTTP 头的整条归档记录。

`-> str` 表示返回 Python 字符串。本函数一直返回字符串；空页面返回空字符串，不用 None 表示空结果。

## 4. 第 9–10 行：先尝试最常见的 UTF-8

```python
try:
    html = html_bytes.decode("utf-8")
```

可以把 decode 理解成“按照指定字典，把字节翻译成文字”。

如果原始内容是 UTF-8 的 `<p>你好</p>`，这句执行成功，html 就是 `"<p>你好</p>"`，直接进入提取步骤，不再调用编码检测器。

这里没有加 `errors="ignore"` 或 `errors="replace"`：我们希望错误编码真正报错，才知道应该走检测分支。如果第一次就吞掉错误，可能得到乱码却永远不检测。

## 5. 第 11–15 行：UTF-8 读不通，再检测

```python
except UnicodeDecodeError:
    encoding = detect_encoding(html_bytes, from_html_meta=True)
    if not encoding:
        encoding = "utf-8"
```

`UnicodeDecodeError` 表示这些字节不能严格按 UTF-8 解释，不是“网页正文提取失败”。

`from_html_meta=True` 让检测器先参考网页里可读的编码声明，例如 `<meta charset="gb18030">`；没有可用声明再自动判断。encoding 是一个名字，比如 `"gb18030"`，不是解码后的正文。

`if not encoding` 是一个简单防守：没有检测结果时仍提供一个编码名。在本次库默认配置下，检测失败通常已经返回 UTF-8 兜底。

注意它是启发式判断，不可能保证猜对所有网页。短文本、错误 charset 声明、混合编码都可能误导检测器；本函数也没有实现完整浏览器级别的编码嗅探流程。

## 6. 第 17 行：按检测结果重新读

```python
html = html_bytes.decode(encoding, errors="replace")
```

如果检测为 GB18030，就按 GB18030 解码。少量仍无法解释的损坏字节，用 `�` 替代，避免因一个坏字符就丢掉整张网页。

这里的 replace 不负责修复事实、语病或 SEO 拼接，只负责在最后一次解码时保留能读出的文字。第一轮严格检查，第二轮容忍局部损坏，两者用途不同。

## 7. 第 19–20 行：提取可见文本并返回

```python
text = extract_plain_text(html, main_content=False, links=False)
return text
```

`html` 是刚解码好的字符串。解析器移除标签表示、脚本和样式等不可见内容，再返回文本，并保留基本段落、列表格式。

`links=False` 表示保留链接文字，但不额外把目标 URL 拼进正文。例如 `<a href="/news">新闻</a>` 仍有“新闻”，不把 `/news` 作为文字追加进去。它不是删除所有含链接的内容。

**`main_content=False` 是重要选择。** 这里执行基础的可见文本提取，因此导航、页脚、标签列表等可见文字可能保留；它不是已经完成正文筛选的清洗器。指定测试的参考输出保留了文章末尾的 Novel 标签列表，基础提取与这个约定一致。

若后续需要尝试去导航/页脚，可以另做 `main_content=True` 实验，但这是启发式正文抽取，也可能误删内容；不能不做观察就承诺更好。当前实现没有针对第一个网页写手工删除规则，也没有针对测试 fixture 做内容特判。

参见官方 [HTML2Text 文档](https://resiliparse.chatnoir.eu/en/stable/api/extract/html2text.html) 与 [编码检测文档](https://resiliparse.chatnoir.eu/en/stable/api/parse/encoding.html)。

## 8. adapter 就是把测试转到实现

```python
def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    from cs336_data.extract import extract_text_from_html_bytes

    return extract_text_from_html_bytes(html_bytes)
```

测试调用 adapter，adapter 调我们的实现，再把结果原样返回。adapter 的类型允许 None，但我们的实现选择返回字符串，不在这里重复编码处理或 HTML 提取。

## 9. 测试验证了什么

原有 `tests/test_extract.py` 检查 Moby-Dick HTML 的输出是否与参考文本完全一致，包括段落和列表格式。

新增 `tests/test_extract_encodings.py` 检查 UTF-8、Windows-1252、GB18030、Shift-JIS，检查可见导航仍保留、脚本被删除、链接 URL 不额外输出，以及空输入返回空字符串。

2026-10-17 运行结果：**7 passed，20 deselected**。分别用项目锁文件版本 Resiliparse/FastWARC 0.15.2 和当前 1.0.9 验证，两组均通过。

当前 assignment4 没有自己的已建项目环境。为避免安装与本题无关的完整训练依赖栈，本次使用 uv 隔离环境，实际命令为：

```bash
UV_CACHE_DIR=/tmp/cs336-uv-cache uv run --no-project \
  --python /Users/lwx/cs336/assignment2-systems/.venv/bin/python \
  --with 'resiliparse==0.15.2' --with 'fastwarc==0.15.2' \
  --with pytest --with xopen \
  python -m pytest -k test_extract_text_from_html_bytes
```

`--python` 借用已有 Python 解释器，但补充的依赖位于隔离环境，不修改 assignment2 的环境。`xopen` 是其他测试模块在收集阶段需要导入的依赖。没有修改 pyproject.toml 或 uv.lock。

正常完整安装 assignment4 环境后，题目原命令是：

```bash
uv run pytest -k test_extract_text_from_html_bytes
```

## 10. 比较脚本怎么处理 WARC 与 WET

脚本按以下顺序操作：

1. 用 FastWARC 读取 WET conversion 记录，按 `WARC-Refers-To` 建立索引，保留 URL 和纯文本。
2. 遍历整份 WARC 中的 response，筛选 HTML/XHTML 响应。
3. 开启 HTTP 内容/传输解码，reader 取出的正文才交给提取函数，不把协议头当正文。
4. 用 response 的 `WARC-Record-ID` 找对应 WET，并核对 URL 一致。
5. 写出新的提取文本，统计匹配记录和字符数，保留几个页面的两种文本供人工对读。

**不能直接用“WARC 第 k 条 = WET 第 k 条”来配对。** WARC 中有 request、metadata、warcinfo 和非 HTML 响应，WET 是经过转换的记录集合。通过记录 ID 配对，才能确认真的是同一页。

脚本流式读取 WARC，但为便于阅读，先把一份 WET 索引保存在内存；它适用于这份单文件实验，不是已经完成的大规模分布式处理管线。字符数变化不等于质量变化，最终优劣仍需对读文本。

复现比较时，把上面隔离命令最后一行换成：

```bash
python -m scripts.compare_html_extraction
```

默认处理整个下载文件，不是只取第一条。试跑可加 `--limit 25 --output data/html_text_comparison_preview`。

结果目录是 `data/html_text_comparison/`：`extracted.jsonl.gz` 为完整新提取结果，`summary.json` 为统计，`samples.json` 为样本对照，`first_resiliparse.txt` 和 `first_wet.txt` 可以直接打开并排看。

## 11. 实际观察与 (b) 回答

2026-10-17 使用锁文件版本 0.15.2 处理整份 WARC，`--limit 0`，结果如下：

- HTML/XHTML 响应 19,871 条，成功提取 19,871 条，提取异常 0 条。
- WET conversion 记录 19,637 条，其中 19,633 条与本次 HTML 响应按 ID 和 URL 匹配。
- 546 条提取结果为空白；“没有抛异常”不等于每条都得到有用内容。
- 匹配样本新文本共 172,183,431 个 Python 字符，WET 共 134,996,829 个，前者约多 27.5%。字符数包含换行、空格、列表符号，不能直接当正文内容增益。

第一张百合花页面：新提取结果 1,694 字符，WET 1,599 字符。主要正文都保留；新文本有更多段落空行和列表项目符号，还保留了网站 logo 的替代文字。两者都保留导航、占位地址、重复电话和页脚，也都保留原 HTML 已有的断句、异常插词和重复问答。

另一张企业新闻页：新结果 1,830 字符，WET 1,465 字符。新结果较清楚地表现了多层导航列表和段落，但不是噪声减少。艺术作品页则从 WET 的 1,451 字符变成 1,423 字符，保留作者简介、媒材等信息，同时省掉部分登录表单字段；仍有重复菜单和登录区标题。

因此不能说“Resiliparse 在所有页面都更干净”。本函数配置在段落/列表可读性上有优势，部分页面少了表单字段，但仍只是基础提取；高质量正文训练样本还需正文筛选和质量过滤。

### (b) 三句话回答

我对整份 WARC 的 HTML 记录运行了提取函数，并按记录 ID 与对应 WET 匹配比较，两者保留的主要正文在人工检查样本中大体一致。Resiliparse 更明确地保留了段落和列表结构，部分页面省掉了表单字段，但第一张网页中两者都保留导航、重复电话、占位地址和页脚，正文原有的异常插词与断句也没有被修复。就所检查页面的阅读排版而言我更偏好 Resiliparse，但当前配置没有明显的正文纯净度优势，不能直接当作清洗完成的训练文本。
