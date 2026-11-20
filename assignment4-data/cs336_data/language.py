"""用 fastText 的 lid.176.bin 识别文本的主要语言。"""

import os
from functools import lru_cache
from pathlib import Path

import fasttext


@lru_cache(maxsize=1)
def _load_language_model(model_path: str):
    # 模型较大，只加载一次；后续调用复用，不重复读取 126 MB 文件。
    return fasttext.load_model(model_path)


def identify_language(text: str) -> tuple[str, float]:
    # fastText 的单条 predict 不接受换行；用空格连接，避免词黏在一起。
    single_line = " ".join(text.split())
    if not single_line:
        return "und", 0.0

    configured_path = os.environ.get("FASTTEXT_LID_MODEL")
    if configured_path:
        model_path = Path(configured_path)
    else:
        model_path = Path("/shared-data/classifiers/lid.176.bin")
        if not model_path.is_file():
            project_dir = Path(__file__).resolve().parents[1]
            model_path = project_dir / "data" / "lid.176.bin"

    if not model_path.is_file():
        raise FileNotFoundError(f"Language model not found: {model_path}. Set FASTTEXT_LID_MODEL to its path.")

    model = _load_language_model(str(model_path.resolve()))
    labels, scores = model.predict(single_line, k=1)
    language = labels[0].removeprefix("__label__")
    # 浮点舍入可能使 fastText 分数略大于 1，约束到接口要求的范围。
    confidence = float(scores[0])
    confidence = max(0.0, min(1.0, confidence))
    return language, confidence
