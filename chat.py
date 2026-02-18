import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI, APIError, AuthenticationError, RateLimitError

load_dotenv()

SYSTEM_PROMPT = """You are a helpful, concise, and friendly assistant.
You answer questions clearly and honestly. When you don't know something,
you say so. You keep responses focused and avoid unnecessary padding. Your name is Selim."""

LOG_FILE = "chat.log"
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# kayıt tutma 

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("llm-cli")
    logger.setLevel(logging.DEBUG)

    # File handler — full debug info
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler — warnings and above only
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()


# groqa bağlanma


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment.")
        print("\n❌  OPENAI_API_KEY bulunamadı. Lütfen .env dosyanızı kontrol edin.\n")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

# cevabı ekrana yazdırma

def stream_response(client: OpenAI, messages: list[dict]) -> str:
    """Send messages to the API with streaming and print tokens as they arrive."""
    full_response = ""
    try:
        with client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
        ) as stream:
            print("\n🤖 Asistan: ", end="", flush=True)
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                print(token, end="", flush=True)
                full_response += token
            print("\n")  # newline after streamed response
    except AuthenticationError:
        logger.error("Authentication failed — invalid API key.")
        print("\n❌  API anahtarı geçersiz. Lütfen kontrol edin.\n")
    except RateLimitError:
        logger.error("Rate limit exceeded.")
        print("\n⚠️  Rate limit aşıldı. Lütfen bir süre bekleyin.\n")
    except APIError as e:
        logger.error("API error: %s", e)
        print(f"\n❌  API hatası: {e}\n")
    return full_response


def print_banner():
    print("=" * 55)
    print("  🤖  LLM CLI Chat  —  powered by OpenAI")
    print(f"  Model : {MODEL}")
    print(f"  Log   : {LOG_FILE}")
    print("  Çıkmak için 'exit' veya 'quit' yazın.")
    print("=" * 55)
    print()


# Main 

def main():
    print_banner()

    client = get_client()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    logger.info("Session started. Model: %s", MODEL)

    while True:
        try:
            user_input = input("💬 Siz: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋  Görüşmek üzere!\n")
            logger.info("Session ended by user (interrupt).")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("\n👋  Görüşmek üzere!\n")
            logger.info("Session ended by user (exit command).")
            break

        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        logger.debug("User: %s", user_input)

        # Get and stream the response
        assistant_reply = stream_response(client, messages)

        if assistant_reply:
            # Add assistant reply to history so context is preserved
            messages.append({"role": "assistant", "content": assistant_reply})
            logger.debug("Assistant: %s", assistant_reply.strip())


if __name__ == "__main__":
    main()
