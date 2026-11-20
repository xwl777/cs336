"""精确行去重，以及基于 MinHash + LSH 的近似文档去重。"""

from collections import Counter
from functools import lru_cache
from hashlib import blake2b
import os
from pathlib import Path
import random
import shutil
import unicodedata


def _line_hash(line: str) -> bytes:
    # 换行符是行的分隔标记，不算正文；保留正文中的大小写和空格。
    content = line.rstrip("\r\n")
    return blake2b(content.encode("utf-8"), digest_size=16).digest()


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
) -> None:
    paths = [Path(path) for path in input_files]
    output_dir = Path(output_directory)

    # 同名输入会写到同一个输出文件，不能悄悄覆盖。
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Input files must have distinct basenames.")
    input_paths = {path.resolve() for path in paths}
    for path in paths:
        if (output_dir / path.name).resolve() in input_paths:
            raise ValueError("Output files must not overwrite input files.")

    # 第一遍：统计全局出现次数。字典只存固定长度哈希，不存整行正文。
    counts = Counter()
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as source:
            for line in source:
                counts[_line_hash(line)] += 1

    # 第二遍：只保留出现一次的行，维持原来的顺序和换行格式。
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as source:
            with (output_dir / path.name).open("w", encoding="utf-8", newline="") as target:
                for line in source:
                    if counts[_line_hash(line)] == 1:
                        target.write(line)


def _normalize_document(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    characters = []
    for character in text:
        # NFD 把 é 拆成 e 和重音符；去掉重音符以及标点。
        if unicodedata.category(character).startswith(("M", "P")):
            continue
        characters.append(character)
    return " ".join("".join(characters).split())


def _document_ngrams(path: Path, n: int) -> set[tuple[str, ...]]:
    words = _normalize_document(path.read_text(encoding="utf-8")).split()
    if len(words) < n:
        # 短文章用整个词序列表示，避免不同短文章都变成同一个空集合。
        # 空文档用 {()} 表示，可以与其他空文档去重。
        return {tuple(words)}
    return {tuple(words[index:index + n]) for index in range(len(words) - n + 1)}


def _minhash_signature(ngrams: set[tuple[str, ...]], num_hashes: int) -> list[int]:
    signature = [2**64 - 1] * num_hashes
    for ngram in ngrams:
        data = " ".join(ngram).encode("utf-8")
        for seed in range(num_hashes):
            # 相同算法配不同 seed，得到不同的哈希函数。
            digest = blake2b(data, digest_size=8, person=seed.to_bytes(8, "little")).digest()
            value = int.from_bytes(digest, "little")
            signature[seed] = min(signature[seed], value)
    return signature


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
) -> None:
    """每个输入文件是一篇文档；每个近似重复的连通分量只输出一篇。"""
    if num_hashes <= 0 or num_bands <= 0 or num_hashes % num_bands != 0:
        raise ValueError("num_hashes must be positive and divisible by positive num_bands.")
    if ngrams <= 0:
        raise ValueError("ngrams must be positive.")
    if not 0 <= jaccard_threshold <= 1:
        raise ValueError("jaccard_threshold must be between 0 and 1.")
    paths = [Path(path) for path in input_files]
    output_dir = Path(output_directory)
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Input files must have distinct basenames.")
    input_paths = {path.resolve() for path in paths}
    if any((output_dir / path.name).resolve() in input_paths for path in paths):
        raise ValueError("Output files must not overwrite input files.")

    # 仅缓存少量文档的词组集合，不把整个语料的所有词组常驻内存。
    @lru_cache(maxsize=16)
    def read_ngrams(document_id):
        return _document_ngrams(paths[document_id], ngrams)

    parent = list(range(len(paths)))
    group_size = [1] * len(paths)

    def find(document_id):
        while parent[document_id] != document_id:
            parent[document_id] = parent[parent[document_id]]
            document_id = parent[document_id]
        return document_id

    def merge(first, second):
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if group_size[first_root] < group_size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        group_size[first_root] += group_size[second_root]

    rows_per_band = num_hashes // num_bands
    buckets = {}
    for document_id in range(len(paths)):
        current = read_ngrams(document_id)
        signature = _minhash_signature(current, num_hashes)
        candidates = set()
        for band in range(num_bands):
            start = band * rows_per_band
            # 带上 band 编号：不同位置的相同片段不算匹配。
            key = (band, tuple(signature[start:start + rows_per_band]))
            bucket = buckets.setdefault(key, [])
            candidates.update(bucket)
            bucket.append(document_id)

        for previous_id in sorted(candidates):
            previous = read_ngrams(previous_id)
            similarity = len(current & previous) / len(current | previous)
            if similarity > jaccard_threshold:
                merge(document_id, previous_id)

    # A≈B、B≈C 时三者归为同一组，即使 A 与 C 没直接匹配。
    groups = {}
    for document_id in range(len(paths)):
        groups.setdefault(find(document_id), []).append(document_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)  # 固定种子，组内均匀抽一篇，方便复现。
    for group in groups.values():
        selected = paths[rng.choice(group)]
        # 比较时归一化；输出保留原始文件字节。
        shutil.copyfile(selected, output_dir / selected.name)
