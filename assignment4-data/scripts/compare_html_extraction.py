"""提取一整个 WARC 的 HTML，并按记录 ID 对照对应 WET，而不是按行号硬配。"""

import argparse
import gzip
import json
from importlib.metadata import version
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.extract import extract_text_from_html_bytes


def load_wet_records(wet_path):
    records = {}
    with open(wet_path, "rb") as stream:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):
            response_id = record.headers.get("WARC-Refers-To")
            url = record.headers.get("WARC-Target-URI")
            text = record.reader.read().decode("utf-8", errors="replace")
            records[response_id] = {"url": url, "text": text}
    return records


def compare(warc_path, wet_path, output_dir, limit):
    output_dir.mkdir(parents=True, exist_ok=True)
    wet_records = load_wet_records(wet_path)
    summary = {
        "warc": str(warc_path),
        "wet": str(wet_path),
        "resiliparse_version": version("resiliparse"),
        "fastwarc_version": version("fastwarc"),
        "main_content": False,
        "links": False,
        "limit": limit,
        "wet_records": len(wet_records),
        "html_records": 0,
        "extracted_records": 0,
        "matched_records": 0,
        "empty_extractions": 0,
        "extracted_chars": 0,
        "matched_extracted_chars": 0,
        "matched_wet_chars": 0,
        "errors": [],
    }
    samples = []
    output_path = output_dir / "extracted.jsonl.gz"

    with open(warc_path, "rb") as stream:
        with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=3) as output:
            records = ArchiveIterator(stream, record_types=WarcRecordType.response, auto_decode="all")
            for record in records:
                content_type = record.http_headers.get("Content-Type", "")
                mime_type = content_type.split(";")[0].strip().lower()
                if mime_type not in ("text/html", "application/xhtml+xml"):
                    continue
                if limit and summary["html_records"] >= limit:
                    break

                summary["html_records"] += 1
                index = summary["html_records"]
                response_id = record.headers.get("WARC-Record-ID")
                url = record.headers.get("WARC-Target-URI")
                # 默认已经解析 HTTP 头；reader 读取的是正文，而非整条 WARC。
                html_bytes = record.reader.read()
                try:
                    text = extract_text_from_html_bytes(html_bytes)
                except (UnicodeError, LookupError, ValueError) as error:
                    summary["errors"].append({"url": url, "error": str(error)})
                    continue

                summary["extracted_records"] += 1
                summary["extracted_chars"] += len(text)
                if not text.strip():
                    summary["empty_extractions"] += 1

                wet_record = wet_records.get(response_id)
                if wet_record is not None:
                    if wet_record["url"] != url:
                        raise ValueError(f"WARC/WET URL mismatch for {response_id}")
                    summary["matched_records"] += 1
                    summary["matched_extracted_chars"] += len(text)
                    summary["matched_wet_chars"] += len(wet_record["text"])

                row = {"record_id": response_id, "url": url, "text": text}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")

                # 保留几个先前人工查看过的非色情页面，便于直接对读。
                if index in (1, 5, 15) and wet_record is not None:
                    samples.append({"html_record_number": index, "url": url,
                                    "resiliparse_text": text, "wet_text": wet_record["text"]})
                if index % 2000 == 0:
                    print(f"Processed {index} HTML records", flush=True)

    if summary["matched_wet_chars"]:
        summary["matched_char_ratio"] = summary["matched_extracted_chars"] / summary["matched_wet_chars"]
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    if samples:
        (output_dir / "first_resiliparse.txt").write_text(samples[0]["resiliparse_text"], encoding="utf-8")
        (output_dir / "first_wet.txt").write_text(samples[0]["wet_text"], encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warc", type=Path, default=Path("data/example.warc.gz"))
    parser.add_argument("--wet", type=Path, default=Path("data/example.warc.wet.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/html_text_comparison"))
    parser.add_argument("--limit", type=int, default=0, help="0 表示处理整个文件；正数用于小样本试跑。")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    compare(args.warc, args.wet, args.output, args.limit)


if __name__ == "__main__":
    main()
