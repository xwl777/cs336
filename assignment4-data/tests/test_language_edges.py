import pytest

from .adapters import run_identify_language


@pytest.mark.parametrize("text", ["", " \t ", "\n\r\n"])
def test_identify_language_empty(text):
    assert run_identify_language(text) == ("und", 0.0)


def test_identify_language_multiline():
    text = "This article explains how plants grow in a garden.\n\nWater, light, and soil help the roots develop."
    language, confidence = run_identify_language(text)
    assert language == "en"
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
