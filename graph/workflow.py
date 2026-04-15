from __future__ import annotations

import hashlib
import re
import time
from typing import Dict, List, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from rag.config import settings
from retrievers.hybrid import HybridRetriever
from retrievers.reranker import DocumentReranker
from utils.llm_factory import get_chat_model
from utils.logger import get_logger, log_event
from utils.redact import redact_for_user_display
from utils.schema_guard import PRIVACY_SCHEMA_ANSWER, is_schema_metadata_probe
from utils.tools import calculator_tool, database_schema_tool, web_search_tool


def _has_count_intent(q: str) -> bool:
    return bool(
        re.search(r"\bhow\s+many\b", q)
        or re.search(r"\bcount\b", q)
        or "number of" in q
        or "total number" in q
        or re.search(r"\btotal\b", q)
    )


def _is_addiction_level_count_query(q: str) -> bool:
    """Route addiction severity counts to SQL; do not rely on RAG snippets."""
    if not _has_count_intent(q):
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


def _is_gender_or_record_count_query(q: str) -> bool:
    """Word boundaries — avoids substring bugs (e.g. 'male' inside 'female')."""
    return bool(
        re.search(r"\bfemale\b", q)
        or re.search(r"\bmale\b", q)
        or re.search(r"\bother\b", q)
        or re.search(r"\bgender\b", q)
        or re.search(r"\busers\b", q)
        or re.search(r"\buser\b", q)
        or re.search(r"\brecord\b", q)
        or re.search(r"\brecords\b", q)
    )


def _tool_aggregate_answer(tool_output: str) -> Optional[str]:
    """Return a definitive user answer from DB tool text without calling the LLM (prevents hallucinations)."""
    if not tool_output:
        return None
    m = re.search(r"Exact joined count:\s*(\d+)", tool_output, re.IGNORECASE)
    if m:
        n = m.group(1)
        return (
            f"There are {n} matching records when combining age with that severity level "
            f"(linked across your stored tables; full count, not a sample)."
        )
    m = re.search(
        r"Exact count for addiction level\s+\w+:\s*(\d+)",
        tool_output,
        re.IGNORECASE,
    )
    if m:
        n = m.group(1)
        return (
            f"There are {n} records with that severity level in the stored dataset "
            f"(full count, not a sample)."
        )
    m = re.search(r"Female users count:\s*(\d+)", tool_output, re.IGNORECASE)
    if m:
        return f"There are {m.group(1)} records in that category (full dataset count)."
    m = re.search(r"Male users count:\s*(\d+)", tool_output, re.IGNORECASE)
    if m:
        return f"There are {m.group(1)} records in that category (full dataset count)."
    m = re.search(r"Other gender users count:\s*(\d+)", tool_output, re.IGNORECASE)
    if m:
        return f"There are {m.group(1)} records in that category (full dataset count)."
    m = re.search(r"Total users count:\s*(\d+)", tool_output, re.IGNORECASE)
    if m:
        return f"There are {m.group(1)} profile records in total (full dataset count)."
    m = re.search(
        r"Records count\s*->\s*users:\s*(\d+)\s*,\s*data:\s*(\d+)",
        tool_output,
        re.IGNORECASE,
    )
    if m:
        return (
            f"There are {m.group(1)} records in the first linked set and {m.group(2)} "
            f"in the second (full dataset counts)."
        )
    return None


class GraphState(TypedDict, total=False):
    query: str
    route: Literal["direct", "rag", "tool", "privacy"]
    rewritten_queries: List[str]
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    compressed_context: str
    answer: str
    memory: List[Dict[str, str]]
    tool_output: str
    error: str


class RAGGraphOrchestrator:
    def __init__(self) -> None:
        self.llm = get_chat_model()
        self.retriever = HybridRetriever()
        self.reranker = DocumentReranker()
        self.logger = get_logger("rag_graph")
        self.response_cache: Dict[str, Dict[str, object]] = {}
        self.graph = self._build_graph()

    def refresh_knowledge(self) -> None:
        self.retriever.refresh()

    def _route_query(self, query: str) -> str:
        q = query.lower().strip()
        q = q.replace("’", "'").replace("`", "'")
        if is_schema_metadata_probe(query):
            return "privacy"
        if any(x in q for x in ["calculate", "+", "-", "*", "/", "math"]):
            return "tool"
        if _has_count_intent(q):
            if _is_addiction_level_count_query(q):
                return "tool"
            if _is_gender_or_record_count_query(q):
                return "tool"
        if any(x in q for x in ["today", "latest", "news", "search web"]):
            return "tool"
        if any(x in q for x in ["database", "table", "sql", "schema"]):
            return "tool"
        direct_greetings = {
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "thanks",
            "thank you",
        }
        if q.strip() in direct_greetings:
            return "direct"
        return "rag"

    def router_node(self, state: GraphState) -> GraphState:
        route = self._route_query(state["query"])
        return {"route": route}

    def rewrite_query_node(self, state: GraphState) -> GraphState:
        base_query = state["query"]
        rewritten = [base_query]
        try:
            prompt = (
                "Generate up to 2 alternative retrieval queries.\n"
                "Return each query on a new line with no bullets.\n"
                f"Original query: {base_query}"
            )
            result = self.llm.invoke(prompt).content
            for line in result.splitlines():
                clean = line.strip("- ").strip()
                if clean and clean.lower() != base_query.lower():
                    rewritten.append(clean)
                if len(rewritten) >= settings.multi_query_count:
                    break
        except Exception:
            pass
        return {"rewritten_queries": rewritten[: settings.multi_query_count]}

    def retrieval_node(self, state: GraphState) -> GraphState:
        queries = state.get("rewritten_queries", [state["query"]])
        docs = self.retriever.retrieve(queries)
        return {"retrieved_docs": docs}

    def rerank_node(self, state: GraphState) -> GraphState:
        reranked = self.reranker.rerank(state["query"], state.get("retrieved_docs", []))
        return {"reranked_docs": reranked}

    def compression_node(self, state: GraphState) -> GraphState:
        docs = state.get("reranked_docs", [])
        budget = settings.max_context_chars
        parts = []
        used = 0
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source_name", "unknown")
            block = f"[{i}] source={source}\n{doc.page_content}\n"
            if used + len(block) > budget:
                break
            used += len(block)
            parts.append(block)
        return {"compressed_context": "\n".join(parts)}

    def privacy_node(self, state: GraphState) -> GraphState:
        return {
            "answer": PRIVACY_SCHEMA_ANSWER,
            "tool_output": "",
            "compressed_context": "",
        }

    def _tool_call(self, query: str) -> str:
        q = query.lower()
        if any(x in q for x in ["calculate", "+", "-", "*", "/", "math"]):
            return calculator_tool.invoke(query.replace("calculate", "").strip() or query)
        if any(x in q for x in ["database", "table", "sql", "schema", "how many", "count", "number of", "total"]):
            return database_schema_tool.invoke(query)
        return web_search_tool.invoke(query)

    def generation_node(self, state: GraphState) -> GraphState:
        route = state.get("route", "rag")
        memory_lines = "\n".join(
            [f"{m.get('role','user').upper()}: {m.get('content','')}" for m in state.get("memory", [])]
        )
        tool_output = state.get("tool_output", "")
        context = state.get("compressed_context", "")

        if route == "tool":
            tool_output = self._tool_call(state["query"])
            direct = _tool_aggregate_answer(tool_output)
            if direct is not None:
                return {
                    "answer": redact_for_user_display(direct),
                    "tool_output": tool_output,
                }

        if route == "direct":
            user_prompt = (
                f"Conversation history:\n{memory_lines}\n\n"
                f"User query: {state['query']}\n"
                "Answer clearly. Do not include raw transaction IDs (e.g. TXN...) in your reply."
            )
        else:
            tool_extra = ""
            if route == "tool" and tool_output:
                tool_extra = (
                    "If tool output contains exact numeric counts, use ONLY those numbers. "
                    "Do not cite WHO, global statistics, or any external source.\n"
                )
            user_prompt = (
                "You are a grounded assistant.\n"
                "Use provided context and tool outputs only.\n"
                f"{tool_extra}"
                "If answer is not in context, say you do not know.\n"
                "Cite evidence references like [1], [2] when possible.\n"
                "Never include raw transaction IDs (values like TXN followed by digits) or "
                "transaction_id fields in your reply; describe records in general terms.\n\n"
                f"Conversation history:\n{memory_lines}\n\n"
                f"Context:\n{context or 'No retrieved context.'}\n\n"
                f"Tool output:\n{tool_output or 'No tool output.'}\n\n"
                f"User query: {state['query']}"
            )
        answer = self.llm.invoke(user_prompt).content
        answer = redact_for_user_display(str(answer))
        return {"answer": answer, "tool_output": tool_output}

    def memory_node(self, state: GraphState) -> GraphState:
        memory = state.get("memory", [])
        memory.append({"role": "user", "content": state["query"]})
        memory.append({"role": "assistant", "content": state.get("answer", "")})
        return {"memory": memory[- settings.max_history_turns * 2 :]}

    def fallback_node(self, state: GraphState) -> GraphState:
        # Basic fallback: one retry with smaller prompt when generation fails.
        try:
            answer = self.llm.invoke(f"Answer briefly: {state['query']}").content
            return {"answer": redact_for_user_display(str(answer))}
        except Exception as exc:
            return {"answer": f"I am unable to answer right now. Error: {exc}"}

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("router_node", self.router_node)
        builder.add_node("rewrite_query_node", self.rewrite_query_node)
        builder.add_node("retrieval_node", self.retrieval_node)
        builder.add_node("rerank_node", self.rerank_node)
        builder.add_node("compression_node", self.compression_node)
        builder.add_node("generation_node", self.generation_node)
        builder.add_node("privacy_node", self.privacy_node)
        builder.add_node("memory_node", self.memory_node)
        builder.add_node("fallback_node", self.fallback_node)

        builder.set_entry_point("router_node")
        builder.add_conditional_edges(
            "router_node",
            lambda s: s.get("route", "rag"),
            {
                "direct": "generation_node",
                "rag": "rewrite_query_node",
                "tool": "generation_node",
                "privacy": "privacy_node",
            },
        )
        builder.add_edge("privacy_node", "memory_node")
        builder.add_edge("rewrite_query_node", "retrieval_node")
        builder.add_edge("retrieval_node", "rerank_node")
        builder.add_edge("rerank_node", "compression_node")
        builder.add_edge("compression_node", "generation_node")
        builder.add_edge("generation_node", "memory_node")
        builder.add_edge("memory_node", END)
        builder.add_edge("fallback_node", "memory_node")
        return builder.compile()

    def ask(self, query: str, memory: List[Dict[str, str]]) -> Dict[str, object]:
        cache_key = hashlib.sha256(f"{query}|{memory}".encode("utf-8")).hexdigest()
        now = time.time()
        cached = self.response_cache.get(cache_key)
        if cached and (now - float(cached["ts"])) < settings.response_cache_ttl_seconds:
            return cached["value"]  # type: ignore[return-value]

        initial: GraphState = {"query": query, "memory": memory}
        try:
            result = self.graph.invoke(initial)
        except Exception:
            result = self.fallback_node(initial)
            result = {**initial, **result}
            result = {**result, **self.memory_node(result)}

        log_event(
            self.logger,
            "rag_query",
            {
                "query": query,
                "rewritten_queries": result.get("rewritten_queries", []),
                "retrieved_count": len(result.get("retrieved_docs", [])),
                "reranked_count": len(result.get("reranked_docs", [])),
                "route": result.get("route", "unknown"),
                "answer_preview": redact_for_user_display(str(result.get("answer", "")))[:300],
            },
        )

        response = {
            "answer": result.get("answer", "No answer generated."),
            "memory": result.get("memory", memory),
            "route": result.get("route", "rag"),
            "retrieved_docs": result.get("reranked_docs", []),
            "debug": {
                "rewritten_queries": result.get("rewritten_queries", []),
                "tool_output": result.get("tool_output", ""),
            },
        }
        self.response_cache[cache_key] = {"ts": now, "value": response}
        return response
