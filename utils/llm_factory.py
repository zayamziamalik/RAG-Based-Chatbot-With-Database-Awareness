from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.config import settings


@lru_cache(maxsize=1)
def get_chat_model():
    provider = settings.llm_provider.lower()
    if provider in {"openai", "groq"}:
        kwargs = {
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
        }
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.llm_model, temperature=settings.llm_temperature)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


@lru_cache(maxsize=1)
def get_embeddings():
    provider = settings.embedding_provider.lower()
    if provider == "openai":
        kwargs = {"model": settings.embedding_model}
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return OpenAIEmbeddings(**kwargs)
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")
