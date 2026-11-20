"""随机抽取20条非空网页，独立标注后运行两个分类器进行复核。"""

import argparse
import csv
import gzip
import json
import random
from pathlib import Path

from cs336_data.harmful import classify_nsfw, classify_toxic_speech


def make_sample(source, output):
    rng = random.Random(2026)
    samples = []
    eligible = 0
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not row["text"].strip():
                continue
            eligible += 1
            if len(samples) < 20:
                samples.append(row)
            else:
                index = rng.randrange(eligible)
                if index < 20:
                    samples[index] = row
    for index, row in enumerate(samples, 1):
        row["sample_number"] = index
    result = {"seed": 2026, "eligible_records": eligible, "samples": samples}
    (output / "sample_unlabeled.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


def evaluate(output):
    sample_data = json.loads((output / "sample_unlabeled.json").read_text())
    annotations = json.loads((output / "human_labels.json").read_text())
    samples = sample_data["samples"]
    if len(samples) != len(annotations):
        raise ValueError("Every sampled document needs an independent manual annotation.")
    results = []
    for sample, human in zip(samples, annotations):
        if (sample["sample_number"], sample["url"]) != (human["sample_number"], human["url"]):
            raise ValueError("Sample and annotation do not match.")
        nsfw_label, nsfw_confidence = classify_nsfw(sample["text"])
        toxic_label, toxic_confidence = classify_toxic_speech(sample["text"])
        # 二分类模型：将“返回标签的分数”统一换成“有害这一类的分数”。
        nsfw_score = nsfw_confidence if nsfw_label == "nsfw" else 1 - nsfw_confidence
        toxic_score = toxic_confidence if toxic_label == "toxic" else 1 - toxic_confidence
        row = dict(human, nsfw_label=nsfw_label, nsfw_confidence=nsfw_confidence,
                   toxic_label=toxic_label, toxic_confidence=toxic_confidence,
                   nsfw_score=nsfw_score, toxic_score=toxic_score)
        row["nsfw_correct"] = (nsfw_label == "nsfw") == human["human_nsfw"]
        row["toxic_correct"] = (toxic_label == "toxic") == human["human_toxic"]
        results.append(row)
        print(sample["sample_number"], nsfw_label, round(nsfw_score, 6), toxic_label, round(toxic_score, 6))

    count = len(results)
    manual_count = sum(r["human_nsfw"] or r["human_toxic"] for r in results)
    predicted_count = sum(r["nsfw_label"] == "nsfw" or r["toxic_label"] == "toxic" for r in results)
    checks = []
    for threshold in (0.5, 0.8, 0.9, 0.95):
        rejected = [r for r in results if r["nsfw_score"] >= threshold or r["toxic_score"] >= threshold]
        checks.append({"threshold": threshold, "rejected": len(rejected),
                       "false_positives": sum(not (r["human_nsfw"] or r["human_toxic"]) for r in rejected),
                       "false_negatives": sum((r["human_nsfw"] or r["human_toxic"]) and r not in rejected
                                              for r in results)})
    summary = {"seed": 2026, "sample_size": count,
               "population_nonempty_documents": sample_data["eligible_records"],
               "policy": "document predominantly NSFW/toxic; contextual quotation/jokes noted separately",
               "manual_harmful_count": manual_count, "manual_harmful_fraction": manual_count / count,
               "predicted_harmful_count": predicted_count, "predicted_harmful_fraction": predicted_count / count,
               "errors": [r for r in results if not r["nsfw_correct"] or not r["toxic_correct"]],
               "threshold_checks": checks,
               "limitations": "20 documents from one WARC; no reliable population rate or calibrated optimal threshold"}
    (output / "predictions.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (output / "review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-sample", action="store_true")
    parser.add_argument("--source", type=Path, default=Path("data/html_text_comparison/extracted.jsonl.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/harmful_content_review"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.make_sample:
        make_sample(args.source, args.output)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
