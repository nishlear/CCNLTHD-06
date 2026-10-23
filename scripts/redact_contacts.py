"""Mask recognizable contact details before research CVs enter the model/index.

This is a deterministic contact filter, not a guarantee of full anonymization.
Names, employers, locations and obfuscated contact details can remain.
"""

import re
from collections import Counter

REDACTION_VERSION = "contacts-v1"

_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}(?![\w.-])")
_URL = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\[\]\"']+")
_PROFILE = re.compile(
    r"(?i)(?<![\w@])(?:linkedin\.com|github\.com|gitlab\.com|t\.me|telegram\.me|"
    r"facebook\.com|instagram\.com|twitter\.com|x\.com)/[^\s<>\[\]\"']+"
)
_INTERNATIONAL_PHONE = re.compile(r"(?<![\w+])\+\d[\d ().-]{6,27}\d(?!\w)")
_LABELLED_PHONE = re.compile(
    r"(?i)\b(?:phone|telephone|tel|mobile|whatsapp|viber)\s*[:=]\s*"
    r"(?P<number>\+?\d[\d ().-]{5,27}\d)(?!\w)"
)
_MESSENGER = re.compile(
    r"(?i)\b(?:telegram|skype|discord)\s*[:=]\s*@?[a-z0-9_.+-]{3,}"
)


def redact_contacts(text: str) -> tuple[str, dict[str, int]]:
    """Return filtered text and aggregate replacement counts, never raw matches."""
    counts = Counter()
    for pattern, category, marker in (
        (_EMAIL, "email", "[REDACTED_EMAIL]"),
        (_URL, "url", "[REDACTED_URL]"),
        (_PROFILE, "profile_link", "[REDACTED_URL]"),
        (_MESSENGER, "messenger", "[REDACTED_CONTACT]"),
    ):
        text, count = pattern.subn(marker, text)
        if count:
            counts[category] += count

    def replace_phone(match):
        number = match.groupdict().get("number") or match.group()
        # International-style numbers are capped at 15 digits. This also avoids
        # treating short quantities such as "+5 years" as phone numbers.
        if not 8 <= sum(character.isdigit() for character in number) <= 15:
            return match.group()
        counts["phone"] += 1
        return "[REDACTED_PHONE]"

    text = _LABELLED_PHONE.sub(replace_phone, text)
    text = _INTERNATIONAL_PHONE.sub(replace_phone, text)
    return text, dict(counts)
