import re
from typing import Dict, List


def has_phone_number(text: object) -> int:
    pattern = r"\b[\+]?[0-9][0-9\-]{9,}\b"
    return int(bool(re.search(pattern, str(text))))


def has_link(text: object) -> int:
    pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return int(bool(re.search(pattern, str(text))))


def keywords_flag(text: object) -> int:
    keywords = [
        "मुफ़्त", "दावा", "जीत", "नकद", "प्रस्ताव", "सीमित", "पुरस्कार", "पैसा",
        "मौका", "लिखकर", "लाख", "stop", "हज़ार", "claim", "free",
        "urgent", "act now", "winner",
        'फ्री', 'जल्दी', 'लिमिटेड', 'विजेता', 'इनाम', 'ऑफर', 'कॉल', 'क्लिक', 'लकी',
        'खरीदें', 'बधाई', 'शीघ्र'
    ]
    txt = str(text).lower()
    for kw in keywords:
        if kw.lower() in txt:
            return 1
    return 0


def special_characters(text: object) -> int:
    pattern = r"(?:₹|RS|INR|\$)\s*\d+(?:,\d+)*(?:\.\d{2})?|[!@#$%^&*(),.?\":{}|<>]"
    return int(bool(re.search(pattern, str(text))))


def cash_amount(text: object) -> int:
    cash_keywords = ["1 लाख", "दस लाख", "1 हज़ार", "दस हज़ार", "करोड़", "दस करोड़", "मिलियन", "बिलियन", "सौ", "लाख", "हज़ार"]
    txt = str(text)
    for c in cash_keywords:
        if c in txt:
            return 1
    return 0


def length_sms(text: object) -> int:
    return len(str(text))


def sms_number(text: object) -> int:
    pattern = r"\b[5-9]\d{4,5}\b"
    return int(bool(re.search(pattern, str(text))))


def word_count(text: object) -> int:
    return len(str(text).split())


def compute_features(message: str) -> Dict[str, int]:
    """Return a feature dict in stable order for a single message."""
    return {
        "contains_phone_number": has_phone_number(message),
        "contains_URL_link": has_link(message),
        "Keywords": keywords_flag(message),
        "Special_Characters": special_characters(message),
        "Amount": cash_amount(message),
        "Length": length_sms(message),
        "SMS_Number": sms_number(message),
        "word_count": word_count(message),
    }


def feature_columns() -> List[str]:
    """Return the canonical feature column order as used in training."""
    return [
        "contains_phone_number",
        "contains_URL_link",
        "Keywords",
        "Special_Characters",
        "Amount",
        "Length",
        "SMS_Number",
        "word_count",
    ]
