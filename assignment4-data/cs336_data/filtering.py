"""网页过滤的逐篇逻辑；返回结果和审计信息，不负责多进程调度。"""

from time import perf_counter

from cs336_data.harmful import classify_nsfw, classify_toxic_speech
from cs336_data.language import identify_language
from cs336_data.pii import mask_emails, mask_ips, mask_phone_numbers
from cs336_data.quality import quality_score


STAGES = ["nonempty", "english", "length", "alphabetic", "nsfw", "toxic", "quality", "pii"]


def filter_document(row, settings):
    """按固定顺序筛选一篇文本；失败即停止，避免昂贵的后续推理。"""
    text = row["text"]
    audit = {"document_id": row["document_id"], "url": row["url"],
             "source": row["source"], "original_chars": len(text), "stages": []}

    def record(stage, passed, start):
        audit["stages"].append({"stage": stage, "passed": passed,
                                "seconds": perf_counter() - start})
        if not passed:
            audit["removed_by"] = stage
        return passed

    start = perf_counter()
    if not record("nonempty", bool(text.strip()), start):
        return None, audit

    start = perf_counter()
    language, confidence = identify_language(text)
    audit.update(language=language, language_confidence=confidence)
    if not record("english", language == "en" and confidence >= settings["english_threshold"], start):
        return None, audit

    start = perf_counter()
    words = text.split()
    audit["words"] = len(words)
    if not record("length", settings["min_words"] <= len(words) <= settings["max_words"], start):
        return None, audit

    start = perf_counter()
    ratio = sum(any(character.isalpha() for character in word) for word in words) / len(words)
    audit["alphabetic_fraction"] = ratio
    if not record("alphabetic", ratio >= settings["min_alphabetic_fraction"], start):
        return None, audit

    start = perf_counter()
    label, confidence = classify_nsfw(text)
    audit.update(nsfw_label=label, nsfw_confidence=confidence)
    reject = label == "nsfw" and confidence >= settings["nsfw_threshold"]
    if not record("nsfw", not reject, start):
        return None, audit

    start = perf_counter()
    label, confidence = classify_toxic_speech(text)
    audit.update(toxic_label=label, toxic_confidence=confidence)
    reject = label == "toxic" and confidence >= settings["toxic_threshold"]
    if not record("toxic", not reject, start):
        return None, audit

    start = perf_counter()
    score = quality_score(text)
    audit["quality_score"] = score
    if not record("quality", score >= settings["quality_threshold"], start):
        return None, audit

    start = perf_counter()
    masked, emails = mask_emails(text)
    masked, phones = mask_phone_numbers(masked)
    masked, ips = mask_ips(masked)
    audit["pii_counts"] = {"emails": emails, "phones": phones, "ips": ips}
    audit["modified"] = masked != text
    record("pii", True, start)
    return dict(row, text=masked), audit
