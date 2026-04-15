"""Detect questions that ask to confirm or name DB schema (tables/columns) — user-facing policy."""

from __future__ import annotations

import re

PRIVACY_SCHEMA_ANSWER = (
    "I can’t confirm or discuss whether specific columns, fields, or tables exist by name, "
    "or describe the database layout. Ask about your topic or metrics in general terms instead."
)

# Questions aimed at metadata / naming (not analytic counts).
_SCHEMA_PROBE: list[re.Pattern[str]] = [
    # "is there user table", "is there a users table", etc. (not only "is there a table")
    re.compile(r"\bis\s+there\s+.*\btable\b", re.I),
    re.compile(r"\bis\s+there\s+(a\s+)?(column|field|table)\b", re.I),
    re.compile(
        r"\b(do\s+we|do\s+i|have\s+we|have\s+i)\s+have\s+.*\btable\b",
        re.I,
    ),
    re.compile(r"\b(users?|data)\s+table\b", re.I),
    re.compile(r"\b(are\s+there)\s+(any\s+)?(columns?|fields?|tables?)\b", re.I),
    re.compile(r"\b(column|field|table)\s+(named|called)\b", re.I),
    re.compile(r"\b(name|names)\s+of\s+(the\s+)?(columns?|fields?|tables?)\b", re.I),
    re.compile(
        r"\b(does|do|is|are)\s+(the\s+)?(column|field|table)\b.*\bexist\b",
        re.I,
    ),
    re.compile(r"\bwhat\s+(is|are)\s+the\s+(columns?|fields?)\b", re.I),
    re.compile(r"\blist\s+(all\s+)?(the\s+)?(columns?|fields?|tables?)\b", re.I),
    re.compile(r"\b(describe|show)\s+(the\s+)?(schema|structure)\b", re.I),
    re.compile(r"\b(information_schema|show\s+create\s+table)\b", re.I),
]


def is_schema_metadata_probe(query: str) -> bool:
    """True if the user is asking to confirm or reveal schema/column/table names."""
    q = (query or "").strip()
    if not q:
        return False
    low = q.lower()
    for pat in _SCHEMA_PROBE:
        if pat.search(low):
            return True
    # Explicit “do I have X” naming a snake_case identifier (common column style).
    if re.search(
        r"\b(do\s+i\s+have|have\s+i\s+got|is\s+there)\b.*\b[a-z]+_[a-z0-9_]+\b",
        low,
        re.I,
    ):
        return True
    return False
