from __future__ import annotations

import ast
import operator as op
import re
from typing import Dict

from langchain_core.tools import tool
from sqlalchemy import create_engine, inspect, text

from rag.config import settings


def _count_intent(q: str) -> bool:
    return bool(
        re.search(r"\bhow\s+many\b", q)
        or re.search(r"\bcount\b", q)
        or "number of" in q
        or "total number" in q
        or re.search(r"\btotal\b", q)
    )


def _addiction_count_intent(q: str) -> bool:
    if not _count_intent(q):
        return False
    if re.search(r"\b(severe|moderate|mild)\b", q) and (
        "addiction" in q
        or "addicted" in q
        or "addict" in q
        or "level" in q
    ):
        return True
    if ("addiction" in q or "addicted" in q) and re.search(
        r"\b(severe|moderate|mild|none)\b", q
    ):
        return True
    return False


def _parse_age_filter(q: str) -> tuple[str, int] | None:
    """Return (SQL comparison operator, age) when the query constrains age, e.g. ('<', 30)."""
    q = q.lower()
    m = re.search(
        r"(?:below|under|less\s+than|younger\s+than)\s+(\d+)",
        q,
    )
    if m:
        return ("<", int(m.group(1)))
    m = re.search(r"<\s*(\d+)", q)
    if m:
        return ("<", int(m.group(1)))
    m = re.search(
        r"(?:above|over|more\s+than|older\s+than)\s+(\d+)",
        q,
    )
    if m:
        return (">", int(m.group(1)))
    m = re.search(r">\s*(\d+)", q)
    if m:
        return (">", int(m.group(1)))
    m = re.search(r"\b(\d+)\s*(?:years?\s*old|yo\b)", q)
    if m:
        return ("=", int(m.group(1)))
    return None


def _addiction_severity_level(q: str) -> str | None:
    for cand in ("severe", "moderate", "mild"):
        if cand in q:
            return cand
    if "none" in q and ("addiction" in q or "addicted" in q):
        return "none"
    return None


def _safe_eval(expr: str) -> float:
    allowed = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return allowed[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)


@tool
def calculator_tool(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely."""
    try:
        return str(_safe_eval(expression))
    except Exception as exc:
        return f"Calculator error: {exc}"


@tool
def web_search_tool(query: str) -> str:
    """Search the web for real-time information."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            rows = list(ddgs.text(query, max_results=5))
        return "\n".join([f"- {r.get('title')}: {r.get('href')}" for r in rows]) or "No results."
    except Exception as exc:
        return f"Web search unavailable: {exc}"


@tool
def database_schema_tool(question: str) -> str:
    """Read database schema and sample rows for quick database Q&A (full detail for the model)."""
    try:
        engine = create_engine(settings.database_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        q = question.lower()
        summary: Dict[str, str] = {}
        with engine.connect() as conn:
            if _addiction_count_intent(q):
                level = _addiction_severity_level(q)
                age_f = _parse_age_filter(q)
                if level is not None and age_f is not None:
                    op, age_val = age_f
                    if op == "<":
                        stmt = text(
                            """
                            SELECT COUNT(*) FROM data d
                            INNER JOIN users u ON d.transaction_id = u.transaction_id
                            WHERE LOWER(TRIM(d.addiction_level)) = :lvl AND u.age < :age
                            """
                        )
                    elif op == ">":
                        stmt = text(
                            """
                            SELECT COUNT(*) FROM data d
                            INNER JOIN users u ON d.transaction_id = u.transaction_id
                            WHERE LOWER(TRIM(d.addiction_level)) = :lvl AND u.age > :age
                            """
                        )
                    else:
                        stmt = text(
                            """
                            SELECT COUNT(*) FROM data d
                            INNER JOIN users u ON d.transaction_id = u.transaction_id
                            WHERE LOWER(TRIM(d.addiction_level)) = :lvl AND u.age = :age
                            """
                        )
                    count = conn.execute(
                        stmt,
                        {"lvl": level, "age": age_val},
                    ).scalar()
                    return (
                        f"Exact joined count: {int(count or 0)} "
                        f"(users linked to data on transaction_id; addiction_level={level}; "
                        f"age {op} {age_val}; full dataset)."
                    )
                if level is not None:
                    count = conn.execute(
                        text(
                            "SELECT COUNT(*) FROM data WHERE LOWER(TRIM(addiction_level)) = :lvl"
                        ),
                        {"lvl": level},
                    ).scalar()
                    return (
                        f"Exact count for addiction level {level.capitalize()}: {int(count or 0)} "
                        f"(full table aggregate, not a sample)."
                    )
            if _count_intent(q) and (
                re.search(r"\bfemale\b", q)
                or re.search(r"\bmale\b", q)
                or re.search(r"\bother\b", q)
                or re.search(r"\bgender\b", q)
                or re.search(r"\busers\b", q)
                or re.search(r"\buser\b", q)
                or re.search(r"\brecord\b", q)
                or re.search(r"\brecords\b", q)
            ):
                if "female" in q:
                    count = conn.execute(
                        text("SELECT COUNT(*) FROM users WHERE LOWER(gender) = 'female'")
                    ).scalar()
                    return f"Female users count: {int(count or 0)}"
                if "male" in q:
                    count = conn.execute(
                        text("SELECT COUNT(*) FROM users WHERE LOWER(gender) = 'male'")
                    ).scalar()
                    return f"Male users count: {int(count or 0)}"
                if "other" in q:
                    count = conn.execute(
                        text("SELECT COUNT(*) FROM users WHERE LOWER(gender) = 'other'")
                    ).scalar()
                    return f"Other gender users count: {int(count or 0)}"
                if "user" in q and "record" not in q and "records" not in q:
                    count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
                    return f"Total users count: {int(count or 0)}"
                if "record" in q or "records" in q:
                    users_count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
                    data_count = conn.execute(text("SELECT COUNT(*) FROM data")).scalar()
                    return (
                        f"Records count -> users: {int(users_count or 0)}, "
                        f"data: {int(data_count or 0)}"
                    )
            for table in tables:
                sample = conn.execute(text(f"SELECT * FROM {table} LIMIT 3")).mappings().all()
                summary[table] = str(sample)
        return f"Question: {question}\nDatabase summary: {summary}"
    except Exception as exc:
        return f"Database tool unavailable: {exc}"
