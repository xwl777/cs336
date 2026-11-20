"""分批并行过滤 WET，合并时全局去重，输出统计和可复核的随机样本。"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
from time import perf_counter

from fastwarc.warc import ArchiveIterator, WarcRecordType

from cs336_data.filtering import STAGES, filter_document


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_priority(document_id):
    # 给每篇文档一个固定的伪随机排序值，合并分片后仍能均匀抽样。
    return hashlib.blake2b(f"42:{document_id}".encode(), digest_size=16).hexdigest()


def add_sample(samples, row, audit, limit=5):
    sample = {"priority": sample_priority(row["document_id"]), "document": row, "audit": audit}
    samples.append(sample)
    samples.sort(key=lambda item: item["priority"])
    del samples[limit:]


def prepare_chunks(inputs, output, chunk_size):
    """流式读取 WET；一个大文件也拆成多个任务，不把全部语料读入内存。"""
    chunks = []
    total = 0
    stream = None
    try:
        for input_path in inputs:
            with input_path.open("rb") as wet:
                records = ArchiveIterator(wet, record_types=WarcRecordType.conversion)
                for record in records:
                    if total % chunk_size == 0:
                        if stream is not None:
                            stream.close()
                        path = output / "chunks" / f"{len(chunks):05d}.jsonl.gz"
                        chunks.append(path)
                        stream = gzip.open(path, "wt", encoding="utf-8", compresslevel=1)
                    row = {"document_id": total, "source": str(input_path.resolve()),
                           "url": record.headers.get("WARC-Target-URI", ""),
                           "record_id": record.headers.get("WARC-Record-ID", ""),
                           "text": record.reader.read().decode("utf-8", errors="replace")}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total += 1
    finally:
        if stream is not None:
            stream.close()
    return chunks, total


def process_chunk(task):
    """子进程处理一个分片；模型由已有函数懒加载，并在进程内缓存复用。"""
    input_path, output, settings = task
    start = perf_counter()
    kept_path = output / "parts" / input_path.name
    audit_path = output / "audit" / input_path.name
    counts = {stage: {"entered": 0, "kept": 0, "removed": 0, "worker_seconds": 0.0} for stage in STAGES}
    removed_samples, modified_samples = [], []
    by_stage = {stage: [] for stage in STAGES}
    pii_counts = Counter()
    modified_documents = 0
    documents = 0
    with gzip.open(input_path, "rt", encoding="utf-8") as source:
        with gzip.open(kept_path, "wt", encoding="utf-8", compresslevel=1) as kept:
            with gzip.open(audit_path, "wt", encoding="utf-8", compresslevel=1) as audits:
                for line in source:
                    row = json.loads(line)
                    result, audit = filter_document(row, settings)
                    documents += 1
                    for step in audit["stages"]:
                        count = counts[step["stage"]]
                        count["entered"] += 1
                        count["kept"] += step["passed"]
                        count["removed"] += not step["passed"]
                        count["worker_seconds"] += step["seconds"]
                    if result is None:
                        add_sample(removed_samples, row, audit)
                        add_sample(by_stage[audit["removed_by"]], row, audit, limit=2)
                    else:
                        # 同行保存处理元数据，合并时不用回读全部 audit。
                        kept.write(json.dumps({"document": result, "audit": audit}, ensure_ascii=False) + "\n")
                        pii_counts.update(audit["pii_counts"])
                        modified_documents += audit["modified"]
                        if audit["modified"]:
                            add_sample(modified_samples, row, dict(audit, masked_text=result["text"]))
                    audits.write(json.dumps(audit, ensure_ascii=False) + "\n")
    return {"kept_path": str(kept_path), "documents": documents, "counts": counts,
            "pii_counts": dict(pii_counts), "modified_documents": modified_documents,
            "removed_samples": removed_samples, "modified_samples": modified_samples,
            "by_stage": by_stage, "worker_elapsed_seconds": perf_counter() - start}


def merge_outputs(results, output):
    """全局去重不能仅在每个进程内部做，否则跨分片重复会漏掉。"""
    seen = set()
    kept_samples, removed_samples, modified_samples = [], [], []
    by_stage = {stage: [] for stage in STAGES + ["exact_document_dedup"]}
    counts = {stage: {"entered": 0, "kept": 0, "removed": 0, "worker_seconds": 0.0} for stage in STAGES}
    dedup = {"entered": 0, "kept": 0, "removed": 0, "worker_seconds": 0.0}
    pii_counts = Counter()
    modified_documents = 0
    final_modified_documents = 0
    characters = 0
    with gzip.open(output / "filtered.jsonl.gz", "wt", encoding="utf-8") as destination:
        with gzip.open(output / "dedup_audit.jsonl.gz", "wt", encoding="utf-8") as audit_stream:
            with gzip.open(output / "training.txt.gz", "wt", encoding="utf-8") as training:
                for result in results:
                    for stage in STAGES:
                        for key in counts[stage]:
                            counts[stage][key] += result["counts"][stage][key]
                    pii_counts.update(result["pii_counts"])
                    modified_documents += result["modified_documents"]
                    removed_samples.extend(result["removed_samples"])
                    for stage in STAGES:
                        by_stage[stage].extend(result["by_stage"][stage])
                    with gzip.open(result["kept_path"], "rt", encoding="utf-8") as source:
                        for line in source:
                            item = json.loads(line)
                            row, audit = item["document"], item["audit"]
                            start = perf_counter()
                            # 在脱敏后的文本上比较；只归一化空白，保留大小写。
                            digest = hashlib.blake2b(" ".join(row["text"].split()).encode(), digest_size=16).digest()
                            duplicate = digest in seen
                            seen.add(digest)
                            dedup["worker_seconds"] += perf_counter() - start
                            dedup["entered"] += 1
                            dedup["removed" if duplicate else "kept"] += 1
                            decision = {"document_id": row["document_id"], "url": row["url"], "duplicate": duplicate}
                            audit_stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
                            if duplicate:
                                audit["removed_by"] = "exact_document_dedup"
                                add_sample(removed_samples, row, audit)
                                add_sample(by_stage["exact_document_dedup"], row, audit, limit=2)
                                continue
                            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                            training.write(row["text"] + "\n\n")
                            characters += len(row["text"])
                            final_modified_documents += audit["modified"]
                            add_sample(kept_samples, row, audit)
                    modified_samples.extend(result["modified_samples"])
    # 分片样本各自保留最小的随机排序值，合并取前5等价于全局取前5。
    for samples in [removed_samples, modified_samples, *by_stage.values()]:
        samples.sort(key=lambda item: item["priority"])
    review = {"seed": 42, "kept": kept_samples, "removed": removed_samples[:5],
              "modified_before_dedup": modified_samples[:5],
              "removed_by_stage": {stage: values[:2] for stage, values in by_stage.items()}}
    write_json(output / "review_samples.json", review)
    counts["exact_document_dedup"] = dedup
    return counts, {"before_dedup_modified_documents": modified_documents,
                    "final_modified_documents": final_modified_documents,
                    "pii_replacements_before_dedup": dict(pii_counts), "final_characters": characters}


def run(inputs, output, settings, workers, chunk_size):
    if output.exists():
        raise FileExistsError(f"Use a new output directory to avoid mixing runs: {output}")
    start = perf_counter()
    for name in ("chunks", "parts", "audit"):
        (output / name).mkdir(parents=True)
    chunks, total = prepare_chunks(inputs, output, chunk_size)
    ingestion_seconds = perf_counter() - start
    print(f"Read {total} WET documents into {len(chunks)} chunks", flush=True)
    filtering_start = perf_counter()
    tasks = [(path, output, settings) for path in chunks]
    # spawn 避免继承其他库的线程状态；每个进程有独立的模型缓存。
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        results = []
        for result in executor.map(process_chunk, tasks):
            results.append(result)
            print(f"Completed {len(results)}/{len(chunks)} chunks", flush=True)
    filtering_seconds = perf_counter() - filtering_start
    merge_start = perf_counter()
    counts, extras = merge_outputs(results, output)
    merge_seconds = perf_counter() - merge_start
    final = counts["exact_document_dedup"]["kept"]
    removed = total - final
    for count in counts.values():
        count["removed_fraction_of_stage_input"] = count["removed"] / count["entered"] if count["entered"] else 0
        count["removed_fraction_of_all_input"] = count["removed"] / total if total else 0
        count["share_of_all_removed"] = count["removed"] / removed if removed else 0
    summary = {"inputs": [str(path.resolve()) for path in inputs], "settings": settings,
               "environment": {"system": platform.system(), "machine": platform.machine(),
                               "python": platform.python_version(), "logical_cpus": os.cpu_count()},
               "input_compressed_bytes": sum(path.stat().st_size for path in inputs),
               "workers": workers, "chunk_size": chunk_size, "chunks": len(chunks),
               "input_documents": total, "kept_documents": final, "removed_documents": removed,
               "retention_fraction": final / total if total else 0, "stages": counts,
               "timing": {"ingestion_wall_seconds": ingestion_seconds,
                          "filtering_wall_seconds": filtering_seconds, "merge_wall_seconds": merge_seconds,
                          "total_wall_seconds": perf_counter() - start}, **extras,
               "notes": ["Stage worker_seconds sum CPU-process wall durations, not end-to-end wall time; includes lazy model loading.",
                         "Exact document dedup keeps the first whitespace-normalized masked document; line and fuzzy dedup not enabled.",
                         "training.txt.gz is readable text with blank separators; use filtered.jsonl.gz for explicit document boundaries.",
                         "No Paloma validation text is used; this run does not demonstrate validation perplexity improvements."]}
    elapsed = summary["timing"]["total_wall_seconds"]
    summary["throughput"] = {"documents_per_second": total / elapsed,
                             "compressed_megabytes_per_second": summary["input_compressed_bytes"] / 1e6 / elapsed}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--english-threshold", type=float, default=0.7)
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--max-words", type=int, default=100_000)
    parser.add_argument("--min-alphabetic-fraction", type=float, default=0.8)
    parser.add_argument("--nsfw-threshold", type=float, default=0.9)
    parser.add_argument("--toxic-threshold", type=float, default=0.9)
    parser.add_argument("--quality-threshold", type=float, default=0.5)
    args = parser.parse_args()
    settings = {key: getattr(args, key) for key in ["english_threshold", "min_words", "max_words",
                "min_alphabetic_fraction", "nsfw_threshold", "toxic_threshold", "quality_threshold"]}
    if args.workers < 1 or args.chunk_size < 1 or not 1 <= args.min_words <= args.max_words:
        parser.error("Positive worker/chunk sizes and 1 <= min_words <= max_words are required.")
    for key in ["english_threshold", "min_alphabetic_fraction", "nsfw_threshold", "toxic_threshold", "quality_threshold"]:
        if not 0 <= settings[key] <= 1:
            parser.error(f"{key} must be between 0 and 1.")
    paths = []
    for path in args.inputs:
        paths.extend(sorted(path.glob("*.warc.wet.gz")) if path.is_dir() else [path])
    if not paths or any(not path.is_file() for path in paths):
        parser.error("Provide existing WET files, or directories containing *.warc.wet.gz.")
    run(paths, args.output, settings, args.workers, args.chunk_size)


if __name__ == "__main__":
    main()
