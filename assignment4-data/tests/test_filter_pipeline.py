import gzip
import json

import pytest

from cs336_data import filtering
from scripts.filter_wet import merge_outputs, process_chunk


SETTINGS = {"english_threshold": 0.7, "min_words": 50, "max_words": 100_000,
            "min_alphabetic_fraction": 0.8, "nsfw_threshold": 0.9,
            "toxic_threshold": 0.9, "quality_threshold": 0.5}


@pytest.fixture
def classifiers(monkeypatch):
    # 用固定分类结果测试流水线控制逻辑，避免单元测试加载数 GB 的模型。
    monkeypatch.setattr(filtering, "identify_language", lambda text: ("en", 0.99))
    monkeypatch.setattr(filtering, "classify_nsfw", lambda text: ("non-nsfw", 0.99))
    monkeypatch.setattr(filtering, "classify_toxic_speech", lambda text: ("non-toxic", 0.99))
    monkeypatch.setattr(filtering, "quality_score", lambda text: 0.8)


def document(index=0, extra=""):
    return {"document_id": index, "source": "fixture.wet.gz", "url": f"https://example.org/{index}",
            "text": "This garden contains flowers and trees. " * 12 + extra}


def test_pipeline_masks_without_rejecting_high_negative_confidence(classifiers):
    result, audit = filtering.filter_document(document(extra="Contact a@example.org"), SETTINGS)
    assert result is not None
    assert "|||EMAIL_ADDRESS|||" in result["text"]
    assert audit["pii_counts"]["emails"] == 1
    assert audit["modified"]
    assert all(step["passed"] for step in audit["stages"])


def test_pipeline_stops_at_first_rejection(classifiers, monkeypatch):
    monkeypatch.setattr(filtering, "identify_language", lambda text: ("en", 0.69))
    result, audit = filtering.filter_document(document(), SETTINGS)
    assert result is None
    assert audit["removed_by"] == "english"
    assert [step["stage"] for step in audit["stages"]] == ["nonempty", "english"]


def test_pipeline_quality_threshold_and_toxic_gate(classifiers, monkeypatch):
    monkeypatch.setattr(filtering, "quality_score", lambda text: 0.49)
    result, audit = filtering.filter_document(document(), SETTINGS)
    assert result is None and audit["removed_by"] == "quality"
    monkeypatch.setattr(filtering, "classify_toxic_speech", lambda text: ("toxic", 0.95))
    result, audit = filtering.filter_document(document(), SETTINGS)
    assert result is None and audit["removed_by"] == "toxic"


def test_pipeline_merges_cross_chunk_duplicates_and_counts(classifiers, tmp_path):
    for name in ("chunks", "parts", "audit"):
        (tmp_path / name).mkdir()
    rows = [document(0, "Contact a@example.org"), document(1, "Contact a@example.org"),
            document(2, "The ocean is blue."), dict(document(3), text=" ")]
    results = []
    for index, batch in enumerate((rows[:1], rows[1:])):
        path = tmp_path / "chunks" / f"{index}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for row in batch:
                stream.write(json.dumps(row) + "\n")
        results.append(process_chunk((path, tmp_path, SETTINGS)))
    counts, extras = merge_outputs(results, tmp_path)
    assert counts["nonempty"]["entered"] == 4
    assert counts["nonempty"]["removed"] == 1
    assert counts["exact_document_dedup"]["entered"] == 3
    assert counts["exact_document_dedup"]["kept"] == 2
    assert counts["exact_document_dedup"]["removed"] == 1
    assert extras["before_dedup_modified_documents"] == 2
    assert extras["final_modified_documents"] == 1
    with gzip.open(tmp_path / "filtered.jsonl.gz", "rt") as stream:
        kept = [json.loads(line) for line in stream]
    assert [row["document_id"] for row in kept] == [0, 2]
    review = json.loads((tmp_path / "review_samples.json").read_text())
    assert {item["audit"]["removed_by"] for item in review["removed"]} == {"nonempty", "exact_document_dedup"}
