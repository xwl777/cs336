"""固定种子抽取20条非空提取文本，再与独立人工标签对照。"""

import argparse
import csv
import gzip
import json
import random
from importlib.metadata import version
from pathlib import Path

from cs336_data.language import identify_language


def make_sample(source, output_dir):
    # 水库抽样：只保留20条，但每条非空记录被选中的机会相同。
    rng = random.Random(42)
    samples = []
    total = 0
    eligible = 0
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            total += 1
            row = json.loads(line)
            if not row["text"].strip():
                continue
            eligible += 1
            row["corpus_record_number"] = total
            if len(samples) < 20:
                samples.append(row)
            else:
                index = rng.randrange(eligible)
                if index < 20:
                    samples[index] = row

    for index, row in enumerate(samples, start=1):
        row["sample_number"] = index
    result = {"seed": 42, "sampling": "uniform reservoir over nonempty extracted documents",
              "total_records": total, "eligible_records": eligible, "samples": samples}
    path = output_dir / "sample_unlabeled.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(samples)} samples to {path}; annotate human_labels.json before evaluation.")


def evaluate(output_dir):
    sample_data = json.loads((output_dir / "sample_unlabeled.json").read_text(encoding="utf-8"))
    human_labels = json.loads((output_dir / "human_labels.json").read_text(encoding="utf-8"))
    samples = sample_data["samples"]
    if len(samples) != len(human_labels):
        raise ValueError("Each sampled document must have one manual label.")

    results = []
    for sample, human in zip(samples, human_labels):
        if sample["url"] != human["url"] or sample["sample_number"] != human["sample_number"]:
            raise ValueError("Manual labels do not match the sample order/URLs.")
        language, confidence = identify_language(sample["text"])
        row = {"sample_number": sample["sample_number"], "url": sample["url"],
               "human_language": human["human_language"], "predicted_language": language,
               "confidence": confidence, "correct": language == human["human_language"],
               "notes": human["notes"]}
        results.append(row)
        print(row["sample_number"], row["human_language"], language, f"{confidence:.6f}", row["correct"])

    english_manual = sum(row["human_language"] == "en" for row in results)
    english_predicted = sum(row["predicted_language"] == "en" for row in results)
    errors = []
    for row in results:
        if not row["correct"]:
            errors.append(row)

    threshold_checks = []
    for threshold in (0.5, 0.8, 0.9, 0.95):
        kept = []
        for row in results:
            if row["predicted_language"] == "en" and row["confidence"] >= threshold:
                kept.append(row)
        true_english_kept = sum(row["human_language"] == "en" for row in kept)
        threshold_checks.append({"threshold": threshold, "kept": len(kept),
                                 "true_english_kept": true_english_kept,
                                 "false_english_kept": len(kept) - true_english_kept,
                                 "manual_english_missed": english_manual - true_english_kept})

    summary = {"seed": sample_data["seed"], "sample_size": len(results),
               "model": "lid.176.bin", "fasttext_version": version("fasttext"),
               "label_policy": "predominant language of extracted text; mixed/boilerplate cases noted separately",
               "population_nonempty_documents": sample_data["eligible_records"],
               "manual_english_count": english_manual,
               "manual_english_fraction": english_manual / len(results),
               "predicted_english_count": english_predicted,
               "predicted_english_fraction": english_predicted / len(results),
               "correct_count": len(results) - len(errors), "errors": errors,
               "english_threshold_checks": threshold_checks}
    (output_dir / "predictions.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-sample", action="store_true")
    parser.add_argument("--source", type=Path, default=Path("data/html_text_comparison/extracted.jsonl.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/language_id_review"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.make_sample:
        make_sample(args.source, args.output)
    else:
        evaluate(args.output)


if __name__ == "__main__":
    main()
