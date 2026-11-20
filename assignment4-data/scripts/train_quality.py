"""按注册域名拆分训练/验证数据，训练一个简单的 fastText 质量分类器。"""

import argparse
from collections import Counter
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import random
from urllib.parse import urlsplit

import fasttext
import tldextract

from cs336_data.quality import normalize_quality_text


_extract_domain = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def hostname(row):
    host = urlsplit(row.get("final_url", row["url"])).hostname or ""
    # 使用库内附带的公共后缀表；例如 news.example.co.uk 归为 example.co.uk。
    return _extract_domain(host).top_domain_under_public_suffix or host


def split_documents(rows):
    train, valid = [], []
    for row in rows:
        # 同一注册域名的所有页面始终进入同一份数据，避免同站模板泄漏。
        digest = hashlib.sha256(("quality-42:" + hostname(row)).encode()).digest()
        if int.from_bytes(digest[:4], "big") % 5 == 0:
            valid.append(row)
        else:
            train.append(row)
    for split in (train, valid):
        if {row["label"] for row in split} != {"wiki", "cc"}:
            raise ValueError("Each split needs both labels; collect more documents.")
    assert not ({hostname(r) for r in train} & {hostname(r) for r in valid})
    random.Random(42).shuffle(train)
    return train, valid


def write_fasttext(rows, path):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            text = normalize_quality_text(row["text"])
            stream.write(f"__label__{row['label']} {text}\n")


def evaluate(model, rows):
    counts = Counter()
    errors = []
    scores = []
    for row in rows:
        labels, probabilities = model.predict(normalize_quality_text(row["text"]), k=2)
        predicted = labels[0].removeprefix("__label__")
        positive_score = dict(zip(labels, probabilities))["__label__wiki"]
        counts[row["label"] + "_as_" + predicted] += 1
        scores.append((row["label"], float(positive_score)))
        if predicted != row["label"]:
            errors.append({"url": row["url"], "expected": row["label"], "predicted": predicted,
                           "confidence": float(probabilities[0])})
    tp, fp = counts["wiki_as_wiki"], counts["cc_as_wiki"]
    fn, tn = counts["wiki_as_cc"], counts["cc_as_cc"]
    checks = []
    for threshold in (0.5, 0.7, 0.9):
        selected = [label for label, score in scores if score >= threshold]
        correct = selected.count("wiki")
        checks.append({"threshold": threshold, "selected": len(selected),
                       "weak_label_precision": correct / len(selected) if selected else 0.0,
                       "weak_label_recall": correct / (tp + fn)})
    return {"documents": len(rows), "labels": dict(Counter(r["label"] for r in rows)),
            "confusion": dict(counts), "weak_label_accuracy": (tp + tn) / len(rows),
            "wiki_precision": tp / (tp + fp) if tp + fp else 0.0,
            "wiki_recall": tp / (tp + fn), "threshold_checks": checks, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/quality"))
    parser.add_argument("--epoch", type=int, default=25)
    args = parser.parse_args()
    # 按实际的换行读取 JSONL；str.splitlines() 会误切文本里的 Unicode 行分隔符。
    with (args.data / "documents.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    train, valid = split_documents(rows)
    write_fasttext(train, args.data / "train.txt")
    write_fasttext(valid, args.data / "valid.txt")
    parameters = dict(lr=0.3, epoch=args.epoch, dim=64, wordNgrams=2, bucket=100_000,
                      minCount=2, loss="softmax", thread=1)
    model = fasttext.train_supervised(input=str(args.data / "train.txt"), **parameters)
    model.save_model(str(args.data / "model.bin"))
    report = {"parameters": parameters,
              "fasttext_version": version("fasttext"),
              "model_sha256": hashlib.sha256((args.data / "model.bin").read_bytes()).hexdigest(),
              "positive_source": "English Wikipedia external links, fetched HTML, English >=0.5, >=100 words",
              "negative_source": "random English documents from the downloaded Common Crawl WARC",
              "split": "sha256 of final registered domain, approximately 80/20; no registered-domain overlap",
              "limitations": "Small weakly labeled corpus: Wikipedia links are not guaranteed high quality, and CC is not guaranteed low quality. Scores are not calibrated quality probabilities.",
              "train": evaluate(model, train), "validation": evaluate(model, valid),
              "train_urls": [r["url"] for r in train], "validation_urls": [r["url"] for r in valid]}
    (args.data / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"train": report["train"]["labels"], "validation": report["validation"]}, indent=2))


if __name__ == "__main__":
    main()
