import pytest

from .adapters import run_mask_emails, run_mask_ips, run_mask_phone_numbers


def test_mask_emails_punctuation_and_repeat():
    text = "<first.last+tag@sub.example.org>, first.last+tag@sub.example.org."
    assert run_mask_emails(text) == ("<|||EMAIL_ADDRESS|||>, |||EMAIL_ADDRESS|||.", 2)


@pytest.mark.parametrize("number", ["+1 (415) 555-0123", "1-415-555-0123", "415.555.0123", "415 555 0123 ext. 42"])
def test_mask_phones_country_and_extension(number):
    assert run_mask_phone_numbers(f"Call {number}.") == ("Call |||PHONE_NUMBER|||.", 1)


def test_mask_phones_does_not_take_part_of_long_number():
    text = "Order 123456789012345; date 2026-10-17."
    assert run_mask_phone_numbers(text) == (text, 0)


def test_mask_ips_boundaries_and_invalid_addresses():
    text = "0.0.0.0 255.255.255.255 256.1.2.3 1.2.3.4.5 999.192.0.2.1"
    expected = "|||IP_ADDRESS||| |||IP_ADDRESS||| 256.1.2.3 1.2.3.4.5 999.192.0.2.1"
    assert run_mask_ips(text) == (expected, 2)


@pytest.mark.parametrize("mask", [run_mask_emails, run_mask_phone_numbers, run_mask_ips])
def test_mask_pii_repeated_application(mask):
    text = "user@example.com +1 415-555-0123 192.0.2.1"
    masked, count = mask(text)
    assert count == 1
    assert mask(masked) == (masked, 0)
    assert mask("") == ("", 0)
