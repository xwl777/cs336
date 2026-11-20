"""用作业指定的 Dolma fastText 模型判断 NSFW 和有毒言论。"""

import os
from functools import lru_cache
from pathlib import Path

import fasttext


@lru_cache(maxsize=2)
def _load_model(path: str):
    # 每个模型约 1 GB，只在第一次使用时加载。
    return fasttext.load_model(path)


def _classify(text: str, kind: str) -> tuple[str, float]:
    if kind == "nsfw":
        positive, negative = "nsfw", "non-nsfw"
        environment_name = "FASTTEXT_NSFW_MODEL"
    else:
        positive, negative = "toxic", "non-toxic"
        environment_name = "FASTTEXT_TOXIC_MODEL"

    single_line = " ".join(text.split())
    # 没有内容就没有判断依据；0 表示没有置信度，而不是确信安全。
    if not single_line:
        return negative, 0.0

    filename = f"dolma_fasttext_{kind}_jigsaw_model.bin"
    configured_path = os.environ.get(environment_name)
    if configured_path:
        path = Path(configured_path)
    else:
        path = Path("/shared-data/classifiers") / filename
        if not path.is_file():
            path = Path(__file__).resolve().parents[1] / "data" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing classifier: {path}. Set {environment_name} to its path.")

    model = _load_model(str(path.resolve()))
    labels, scores = model.predict(single_line, k=1)
    label = labels[0].removeprefix("__label__")
    if label not in (positive, negative):
        raise ValueError(f"Unexpected label {label!r} from {path}")
    confidence = max(0.0, min(1.0, float(scores[0])))
    # confidence 属于返回的标签；non-toxic, 0.99 表示模型认为无毒。
    return label, confidence


def classify_nsfw(text: str) -> tuple[str, float]:
    return _classify(text, "nsfw")


def classify_toxic_speech(text: str) -> tuple[str, float]:
    return _classify(text, "hatespeech")
