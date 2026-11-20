"""整文件屏蔽 PII，用固定种子随机抽取20条实际发生替换的文档供人工检查。"""

import argparse
import gzip
import json
import random
from pathlib import Path

from cs336_data.pii import (
    _EMAIL_PATTERN,
    _IP_PATTERN,
    _PHONE_PATTERN,
    mask_emails,
    mask_ips,
    mask_phone_numbers,
)


def match_contexts(text):
    contexts = []
    stages = [
        ("email", _EMAIL_PATTERN, "|||EMAIL_ADDRESS|||"),
        ("phone", _PHONE_PATTERN, "|||PHONE_NUMBER|||"),
        ("ip", _IP_PATTERN, "|||IP_ADDRESS|||"),
    ]
    for kind, pattern, placeholder in stages:
        for match in pattern.finditer(text):
            start = max(0, match.start() - 70)
            end = min(len(text), match.end() + 70)
            contexts.append({"type": kind, "value": match.group(), "context": text[start:end]})
        # 与正式屏蔽顺序相同，避免把已被屏蔽邮箱中的数字重复算作电话。
        text = pattern.sub(placeholder, text)
    return contexts


def evaluate(source, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    sample = []
    summary = {"seed": 42, "documents": 0, "documents_with_replacements": 0,
               "emails": 0, "phones": 0, "ips": 0}
    output_path = output_dir / "masked.jsonl.gz"
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=3) as output:
            for line in stream:
                row = json.loads(line)
                original = row["text"]
                masked, email_count = mask_emails(original)
                masked, phone_count = mask_phone_numbers(masked)
                masked, ip_count = mask_ips(masked)
                counts = {"emails": email_count, "phones": phone_count, "ips": ip_count}
                summary["documents"] += 1
                for key, count in counts.items():
                    summary[key] += count
                row["text"] = masked
                row["mask_counts"] = counts
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

                if email_count + phone_count + ip_count == 0:
                    continue
                summary["documents_with_replacements"] += 1
                candidate = {"record_id": row["record_id"], "url": row["url"],
                             "original_text": original, "masked_text": masked,
                             "counts": counts}
                if len(sample) < 20:
                    sample.append(candidate)
                else:
                    index = rng.randrange(summary["documents_with_replacements"])
                    if index < 20:
                        sample[index] = candidate

    for index, row in enumerate(sample, start=1):
        row["sample_number"] = index
        # 按相同屏蔽顺序保存实际命中的局部上下文。
        row["match_contexts"] = match_contexts(row["original_text"])
    summary["sample_size"] = len(sample)
    (output_dir / "samples.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/html_text_comparison/extracted.jsonl.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/pii_review"))
    args = parser.parse_args()
    evaluate(args.source, args.output)


if __name__ == "__main__":
    main()
