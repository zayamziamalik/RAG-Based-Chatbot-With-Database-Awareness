from pathlib import Path

from rag.service import ProductionRAGChatbot


def ensure_text_dir_exists() -> None:
    text_dir = Path("data/text_files")
    text_dir.mkdir(parents=True, exist_ok=True)


def run() -> None:
    ensure_text_dir_exists()
    bot = ProductionRAGChatbot()

    print("Context-Aware RAG Chatbot")
    print("Commands: /refresh to reload docs+db, /exit to quit\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() == "/exit":
            break
        if question.lower() == "/refresh":
            bot.refresh_knowledge()
            print("Bot: knowledge base refreshed.\n")
            continue
        answer = bot.ask(question)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    run()
