from __future__ import annotations

import re

# e.g. TXN00001, TXN07499
_TXN_TOKEN = re.compile(r"\bTXN[A-Z0-9]+\b", re.IGNORECASE)
# transaction_id: foo, transaction_id=foo
_TXN_LABELED = re.compile(
    r"\btransaction[_\s]?id\s*[:=]\s*[^\s\]\),;]+",
    re.IGNORECASE,
)

# Dataset column names (hide from end users only; model may see full names internally).
_COLUMN_NAMES: tuple[str, ...] = (
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
    "stress_level",
    "academic_work_impact",
    "addiction_level",
    "addicted_label",
    "transaction_id",
    "user_id",
    "gender",
    "age",
)

_SQL_TABLE_USERS = re.compile(
    r"\b(FROM|JOIN|INTO|UPDATE|TABLE)\s+users\b",
    re.IGNORECASE,
)
_SQL_TABLE_DATA = re.compile(
    r"\b(FROM|JOIN|INTO|UPDATE|TABLE)\s+data\b",
    re.IGNORECASE,
)
_TABLE_LABEL = re.compile(r"\bTable:\s*(users|data)\b", re.IGNORECASE)
_SOURCE_META = re.compile(r"\bsource=(users|data)\b", re.IGNORECASE)
_QUALIFIED = re.compile(r"\b(users|data)\.", re.IGNORECASE)


def redact_transaction_ids_for_display(text: str) -> str:
    """Strip transaction identifiers from user-visible assistant text."""
    if not text:
        return text
    s = _TXN_TOKEN.sub("[redacted]", text)
    s = _TXN_LABELED.sub("transaction_id: [redacted]", s)
    return s


def redact_schema_names_for_display(text: str) -> str:
    """Remove table/column identifiers from user-visible text (model may still use them internally)."""
    if not text:
        return text
    s = _SQL_TABLE_USERS.sub(r"\1 [redacted]", text)
    s = _SQL_TABLE_DATA.sub(r"\1 [redacted]", s)
    s = _TABLE_LABEL.sub("Table: [redacted]", s)
    s = _SOURCE_META.sub("source=[redacted]", s)
    s = _QUALIFIED.sub("[redacted].", s)
    for col in sorted(_COLUMN_NAMES, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(col)}\b", "[field]", s, flags=re.IGNORECASE)
    return s


def redact_for_user_display(text: str) -> str:
    """Full sanitization for text shown to the user."""
    return redact_schema_names_for_display(redact_transaction_ids_for_display(text))
