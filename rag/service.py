from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from graph.workflow import RAGGraphOrchestrator


@dataclass
class ChatMemory:
    messages: List[Dict[str, str]] = field(default_factory=list)

    def clear(self) -> None:
        self.messages.clear()


class ProductionRAGChatbot:
    def __init__(self) -> None:
        self.memory = ChatMemory()
        self.orchestrator = RAGGraphOrchestrator()

    def refresh_knowledge(self) -> None:
        self.orchestrator.refresh_knowledge()

    def ask(self, question: str) -> str:
        result = self.orchestrator.ask(question, self.memory.messages)
        self.memory.messages = result["memory"]  # type: ignore[assignment]
        return str(result["answer"])

    def ask_with_meta(self, question: str) -> Dict[str, object]:
        result = self.orchestrator.ask(question, self.memory.messages)
        self.memory.messages = result["memory"]  # type: ignore[assignment]
        return result
