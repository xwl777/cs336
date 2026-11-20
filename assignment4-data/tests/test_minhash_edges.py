from cs336_data import deduplication
from .adapters import run_minhash_deduplication


def test_minhash_deduplication_candidates_need_jaccard_and_transitive_clusters(tmp_path, monkeypatch):
    # 强制所有文档进同一桶，检验“进桶不等于重复”。
    monkeypatch.setattr(deduplication, "_minhash_signature", lambda grams, count: [0] * count)
    texts = {"a.txt": "a b c", "b.txt": "a b c d", "c.txt": "b c d", "d.txt": "x y z"}
    paths = []
    for name, text in texts.items():
        path = tmp_path / name
        path.write_text(text)
        paths.append(path)
    output = tmp_path / "out"
    run_minhash_deduplication(paths, 4, 2, 1, 0.6, output)
    # A-B 与 B-C 的 Jaccard 都为 .75，A-C 为 .5；仍应合为同一组。
    names = {path.name for path in output.iterdir()}
    assert "d.txt" in names
    assert len(names & {"a.txt", "b.txt", "c.txt"}) == 1
    for path in output.iterdir():
        assert path.read_text() == texts[path.name]


def test_minhash_deduplication_normalization_and_short_documents(tmp_path):
    texts = ["CAFÉ,   NOIR!", "cafe\nnoir", "different", "", "  \n"]
    paths = []
    for index, text in enumerate(texts):
        path = tmp_path / f"{index}.txt"
        path.write_text(text)
        paths.append(path)
    output = tmp_path / "out"
    run_minhash_deduplication(paths, 32, 8, 5, 0.8, output)
    kept = {path.name for path in output.iterdir()}
    assert len(kept & {"0.txt", "1.txt"}) == 1
    assert "2.txt" in kept
    assert len(kept & {"3.txt", "4.txt"}) == 1
    assert len(kept) == 3


def test_minhash_deduplication_requires_strictly_greater_than_threshold(tmp_path):
    paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
    for path in paths:
        path.write_text("identical text")
    output = tmp_path / "out"
    run_minhash_deduplication(paths, 8, 2, 1, 1.0, output)
    assert len(list(output.iterdir())) == 2
