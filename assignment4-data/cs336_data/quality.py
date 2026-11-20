"""预测网页质量：wiki 表示更像可信引用页，cc 表示更像随机网页。"""

import os
from functools import lru_cache
from pathlib import Path

import fasttext


def normalize_quality_text(text: str) -> str:
    # 训练和预测必须用同一种预处理；顺便去掉 fastText 的标签控制串。
    return " ".join(text.replace("__label__", " ").lower().split())


@lru_cache(maxsize=1)
def _load_quality_model(path: str):
    return fasttext.load_model(path)


def _predict_quality(text: str):
    configured = os.environ.get("FASTTEXT_QUALITY_MODEL")
    path = Path(configured) if configured else Path(__file__).resolve().parents[1] / "data/quality/model.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Quality model not found: {path}. Run scripts.train_quality first.")
    model = _load_quality_model(str(path.resolve()))
    labels, scores = model.predict(normalize_quality_text(text), k=2)
    return [(label.removeprefix("__label__"), max(0.0, min(1.0, float(score))))
            for label, score in zip(labels, scores)]


def quality_score(text: str) -> float:
    """始终返回高质量类别 wiki 的分数，数值越大越倾向保留。"""
    if not text.strip():
        return 0.0
    for label, score in _predict_quality(text):
        if label == "wiki":
            return score
    raise ValueError("The quality model must contain the wiki label.")


def classify_quality(text: str) -> tuple[str, float]:
    """返回获胜类别及其置信度；与 quality_score 的含义不同。"""
    if not text.strip():
        return "cc", 0.0
    predictions = _predict_quality(text)
    label, confidence = predictions[0]
    if {label for label, score in predictions} != {"wiki", "cc"}:
        raise ValueError("Expected a binary wiki/cc quality model.")
    return label, confidence
