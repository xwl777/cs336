import pytest

from cs336_data.harmful import classify_nsfw, classify_toxic_speech


@pytest.mark.parametrize("classify, negative", [
    (classify_nsfw, "non-nsfw"),
    (classify_toxic_speech, "non-toxic"),
])
def test_harmful_empty_has_no_confidence(classify, negative):
    assert classify(" \n\t ") == (negative, 0.0)


@pytest.mark.parametrize("classify, negative", [
    (classify_nsfw, "non-nsfw"),
    (classify_toxic_speech, "non-toxic"),
])
def test_harmful_accepts_multiline(classify, negative):
    label, score = classify("The garden has flowers.\nThe trees are green.\tHave a lovely day.")
    assert label == negative
    assert isinstance(score, float)
    assert 0.5 <= score <= 1.0
