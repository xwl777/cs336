"""获取 Wikipedia 外链正例，以及已有 Common Crawl 中的英文负例。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import random
import urllib.parse
import urllib.request

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.language import identify_language


USER_AGENT = "CS336-quality-classifier/1.0 (small educational data collection)"


def read_url(url, limit=2_000_000):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=8) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Response exceeds byte limit")
        return data, response.geturl(), response.headers.get("Content-Type", "")


def discover(output, rounds):
    shared = Path("/shared-data/wiki/enwiki-20260501-extracted_urls.txt.gz")
    links = {}
    if shared.is_file():
        with gzip.open(shared, "rt") as stream:
            for line in stream:
                url = line.strip()
                links[url] = {"url": url, "source": str(shared)}
    else:
        for index in range(rounds):
            query = {"action": "query", "generator": "random", "grnnamespace": 0,
                     "grnlimit": 50, "prop": "extlinks", "ellimit": "max", "format": "json"}
            url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(query)
            raw, _, _ = read_url(url)
            (output / f"wiki_api_{index:02d}.json").write_bytes(raw)
            data = json.loads(raw)
            for page in data.get("query", {}).get("pages", {}).values():
                for entry in page.get("extlinks", []):
                    external = entry["*"]
                    links.setdefault(external, {"url": external, "source": page["title"],
                                               "wikipedia_page_id": page["pageid"]})
            print("Wikipedia batch", index + 1, "unique external links", len(links), flush=True)
    rows = []
    for url, row in links.items():
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            continue
        if parts.hostname.endswith(("wikipedia.org", "wikimedia.org")):
            continue
        if parts.path.lower().endswith((".pdf", ".jpg", ".png", ".zip", ".gz", ".mp3", ".mp4")):
            continue
        rows.append(row)
    random.Random(42).shuffle(rows)
    (output / "positive_urls.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print("Eligible external URLs:", len(rows), flush=True)


def fetch_page(item):
    row, pages = item
    key = hashlib.sha256(row["url"].encode()).hexdigest()
    path = pages / f"{key}.html.gz"
    info_path = pages / f"{key}.json"
    metadata = dict(row)
    try:
        if path.is_file() and info_path.is_file():
            with gzip.open(path, "rb") as stream:
                raw = stream.read()
            metadata["final_url"] = json.loads(info_path.read_text())["final_url"]
        else:
            raw, final_url, content_type = read_url(row["url"])
            if "html" not in content_type.lower():
                raise ValueError(f"Not HTML: {content_type}")
            with gzip.open(path, "wb") as stream:
                stream.write(raw)
            metadata["final_url"] = final_url
            info_path.write_text(json.dumps({"final_url": final_url, "content_type": content_type}))
        metadata["text"] = extract_text_from_html_bytes(raw)
        metadata["html_file"] = str(path)
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
    return metadata


def crawl(output, limit):
    rows = json.loads((output / "positive_urls.json").read_text())[:limit]
    pages = output / "pages"
    pages.mkdir(exist_ok=True)
    successes = 0
    with (output / "crawl.jsonl").open("w", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for index, row in enumerate(pool.map(fetch_page, [(row, pages) for row in rows]), 1):
                successes += "text" in row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if index % 25 == 0:
                    print("Fetched", index, "HTML successes", successes, flush=True)
    print("Crawl finished:", len(rows), "attempted,", successes, "HTML pages", flush=True)


def usable_english(text):
    # 两类都做同样的筛选，避免模型仅凭“是不是英文”来分类。
    if not 100 <= len(text.split()) <= 100_000:
        return False
    beginning = text[:500].lower()
    if any(message in beginning for message in ("just a moment...", "verify you are human",
                                                "access denied", "checking your browser")):
        return False
    language, confidence = identify_language(text)
    return language == "en" and confidence >= 0.5


def prepare(output, negative_source):
    positive = []
    negative = []
    seen = set()
    with (output / "crawl.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            text = row.get("text", "")
            digest = hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
            if digest in seen or not usable_english(text):
                continue
            seen.add(digest)
            positive.append(dict(row, label="wiki", text_sha256=digest))
    with gzip.open(negative_source, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            text = row["text"]
            digest = hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
            if digest in seen or not usable_english(text):
                continue
            seen.add(digest)
            negative.append(dict(row, label="cc", text_sha256=digest))
    eligible_negatives = len(negative)
    rng = random.Random(42)
    rng.shuffle(negative)
    negative = negative[:len(positive)]
    if len(positive) < 50 or len(negative) < 50:
        raise ValueError(f"Too few usable examples: positive={len(positive)}, negative={len(negative)}")
    rows = positive + negative
    rng.shuffle(rows)
    with (output / "documents.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"positive_source": "Wikipedia external links; original API responses retained",
               "negative_source": str(negative_source), "seed": 42,
               "english_min_confidence": 0.5, "word_count_range": [100, 100_000],
               "deduplication": "SHA256 of lowercase whitespace-normalized text across both classes",
               "positive_documents": len(positive), "eligible_cc_documents": eligible_negatives,
               "sampled_cc_documents": len(negative)}
    (output / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Prepared", len(positive), "wiki and", len(negative), "cc examples", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("discover", "crawl", "prepare"))
    parser.add_argument("--output", type=Path, default=Path("data/quality"))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--limit", type=int, default=700)
    parser.add_argument("--negative-source", type=Path, default=Path("data/html_text_comparison/extracted.jsonl.gz"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "discover":
        discover(args.output, args.rounds)
    elif args.stage == "crawl":
        crawl(args.output, args.limit)
    else:
        prepare(args.output, args.negative_source)


if __name__ == "__main__":
    main()
