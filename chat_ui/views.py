from __future__ import annotations

import threading

from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_POST

from rag.service import ProductionRAGChatbot


_BOT: ProductionRAGChatbot | None = None
_REFRESH_LOCK = threading.Lock()
_REFRESH_RUNNING = False


def _get_bot() -> ProductionRAGChatbot:
    global _BOT
    if _BOT is None:
        _BOT = ProductionRAGChatbot()
    return _BOT


def _load_history(request: HttpRequest) -> list[dict[str, str]]:
    return request.session.get("chat_history", [])


def _save_history(request: HttpRequest, history: list[dict[str, str]]) -> None:
    request.session["chat_history"] = history
    request.session.modified = True


def _sync_memory_from_session(request: HttpRequest) -> None:
    # Session history is the source of truth for per-browser memory.
    _get_bot().memory.messages = list(_load_history(request))


@require_GET
def chat_page(request: HttpRequest) -> HttpResponse:
    history = _load_history(request)
    return render(request, "chat_ui/chat.html", {"history": history})


@require_POST
def ask_question(request: HttpRequest) -> HttpResponse:
    question = request.POST.get("question", "").strip()
    if not question:
        return redirect("chat_page")

    history = _load_history(request)
    _sync_memory_from_session(request)
    answer = _get_bot().ask(question)

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    _save_history(request, history[-20:])
    return redirect("chat_page")


@require_POST
def refresh_knowledge(request: HttpRequest) -> HttpResponse:
    global _REFRESH_RUNNING
    history = _load_history(request)

    with _REFRESH_LOCK:
        if _REFRESH_RUNNING:
            history.append(
                {
                    "role": "assistant",
                    "content": "Knowledge refresh is already running in background.",
                }
            )
            _save_history(request, history[-20:])
            return redirect("chat_page")
        _REFRESH_RUNNING = True

    def _run_refresh() -> None:
        global _REFRESH_RUNNING
        try:
            _get_bot().refresh_knowledge()
        finally:
            with _REFRESH_LOCK:
                _REFRESH_RUNNING = False

    threading.Thread(target=_run_refresh, daemon=True).start()
    history.append(
        {
            "role": "assistant",
            "content": "Started knowledge refresh in background. You can continue chatting.",
        }
    )
    _save_history(request, history[-20:])
    return redirect("chat_page")


@require_POST
def clear_chat(request: HttpRequest) -> HttpResponse:
    request.session["chat_history"] = []
    request.session.modified = True
    _get_bot().memory.messages.clear()
    return redirect("chat_page")
