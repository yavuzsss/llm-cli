#!/usr/bin/env python3
"""
Otel Resepsiyon Asistanı — Gelişmiş CLI LLM Uygulaması
- Rezervasyon yapabilme
- Çok dil desteği (TR/EN/DE/FR/AR)
- Konuşma geçmişini dosyaya kaydetme
- Fiyat hesaplama
"""

import os
import sys
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI, APIError, AuthenticationError, RateLimitError

# ---------------------------------------------------------------------------
# Konfigürasyon
# ---------------------------------------------------------------------------

load_dotenv()

HOTEL_DATA = """
HOTEL INFORMATION / OTEL BİLGİLERİ:
- Hotel Name / Otel Adı: Grand Yavuz Hotel
- Address / Adres: Bağdat Caddesi No:42, İstanbul

CHECK-IN / CHECK-OUT:
- Check-in: 14:00
- Check-out: 12:00
- Early check-in: On request, subject to availability (extra charge may apply)
- Late check-out: On request, subject to availability (extra charge may apply)

ROOM TYPES & PRICES (per night) / ODA TİPLERİ VE FİYATLARI (gecelik):
- Standard Room / Standart Oda: 2500 TL — 1 double bed, city view, 25 m²
- Deluxe Room / Deluxe Oda: 3500 TL — 1 king bed, sea view, 35 m²
- Suite: 6000 TL — 1 king bed + living room, panoramic view, 60 m²
- Family Room / Aile Odası: 4500 TL — 2 separate beds, large bathroom, 45 m²

SERVICES / HİZMETLER:
- Free Wi-Fi (all areas)
- Open buffet breakfast (07:00 - 10:30) — included in room price
- Restaurant (12:00 - 22:00)
- Spa & Wellness (09:00 - 21:00)
- Fitness center (24 hours)
- Outdoor pool (08:00 - 20:00, May-October)
- Parking (paid, 150 TL/day)
- Airport transfer (reservation required)
- Room service (24 hours)
- Dry cleaning (09:00 - 18:00)
- Concierge (24 hours)

PAYMENT / ÖDEME:
- Cards: Visa, Mastercard, American Express
- Cash accepted
- Deposit: 500 TL at check-in

CANCELLATION / İPTAL POLİTİKASI:
- Free cancellation up to 48 hours before arrival
- Within 48 hours: 1 night charge applies

CONTACT / İLETİŞİM:
- Phone: +90 212 555 00 42
- Email: info@grandyavuzhotel.com
- Reception: 24/7
"""

SYSTEM_PROMPT = f"""You are the professional and friendly reception assistant of Grand Yavuz Hotel.

LANGUAGE RULE: Detect the language of the guest's message and always reply in that same language.
Supported languages: Turkish, English, German, French, Arabic.

RESERVATION RULE: When a guest wants to make a reservation, collect these details one by one:
1. Full name
2. Room type (Standard / Deluxe / Suite / Family)
3. Check-in date
4. Check-out date
5. Number of guests
Then confirm the reservation and calculate the total price based on number of nights.

PRICE CALCULATION: Calculate total = room price per night × number of nights.
Always show the breakdown clearly.

RULES:
- Keep answers short and friendly
- Only help with hotel-related topics
- Never make up information not in the hotel data
- If you don't know something, direct the guest to the front desk

HOTEL DATA:
{HOTEL_DATA}"""

LOG_FILE = "hotel_chat.log"
RESERVATIONS_FILE = "reservations.json"
HISTORY_FILE = "conversation_history.json"
MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("MODEL_API_BASE_URL", "https://api.groq.com/openai/v1")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("hotel-cli")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()

# ---------------------------------------------------------------------------
# Konuşma Geçmişi
# ---------------------------------------------------------------------------

def save_conversation(messages: list[dict], session_id: str):
    """Konuşma geçmişini JSON dosyasına kaydeder."""
    history = []

    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    # System prompt hariç kaydet
    conversation = {
        "session_id": session_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": [m for m in messages if m["role"] != "system"]
    }

    history.append(conversation)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    logger.info("Konuşma geçmişi kaydedildi: %s", HISTORY_FILE)


def save_reservation(reservation_text: str, session_id: str):
    """Rezervasyon bilgilerini JSON dosyasına kaydeder."""
    reservations = []

    if os.path.exists(RESERVATIONS_FILE):
        try:
            with open(RESERVATIONS_FILE, "r", encoding="utf-8") as f:
                reservations = json.load(f)
        except Exception:
            reservations = []

    reservation = {
        "session_id": session_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "details": reservation_text
    }

    reservations.append(reservation)

    with open(RESERVATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(reservations, f, ensure_ascii=False, indent=2)

    logger.info("Rezervasyon kaydedildi: %s", RESERVATIONS_FILE)

# ---------------------------------------------------------------------------
# Yardımcı Fonksiyonlar
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    if not API_KEY:
        logger.error("OPENAI_API_KEY bulunamadı.")
        print("\n❌  API anahtarı bulunamadı. Lütfen .env dosyanızı kontrol edin.\n")
        sys.exit(1)
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def stream_response(client: OpenAI, messages: list[dict]) -> str:
    full_response = ""
    try:
        with client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
        ) as stream:
            print("\n🏨 Resepsiyon: ", end="", flush=True)
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                print(token, end="", flush=True)
                full_response += token
            print("\n")
    except AuthenticationError:
        logger.error("API anahtarı geçersiz.")
        print("\n❌  API anahtarı geçersiz. Lütfen .env dosyanızı kontrol edin.\n")
    except RateLimitError:
        logger.error("Rate limit aşıldı.")
        print("\n⚠️  Çok fazla istek gönderildi. Lütfen bekleyin.\n")
    except APIError as e:
        logger.error("API hatası: %s", e)
        print(f"\n❌  Bir hata oluştu: {e}\n")
    return full_response


def print_banner(session_id: str):
    print("=" * 55)
    print("  🏨  Grand Yavuz Hotel — Resepsiyon Asistanı")
    print(f"  Model   : {MODEL}")
    print(f"  API     : {API_BASE_URL}")
    print(f"  Oturum  : {session_id}")
    print("  Çıkmak için 'exit' veya 'quit' yazın.")
    print("  Rezervasyon listesi için 'rezervasyonlar' yazın.")
    print("=" * 55)
    print("\nHoş geldiniz! / Welcome! / Willkommen!\n")

# ---------------------------------------------------------------------------
# Ana Döngü
# ---------------------------------------------------------------------------

def main():
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print_banner(session_id)

    client = get_client()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    logger.info("Oturum başladı. ID: %s | Model: %s | API: %s", session_id, MODEL, API_BASE_URL)

    while True:
        try:
            user_input = input("💬 Misafir: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋  İyi günler dileriz! / Goodbye!\n")
            save_conversation(messages, session_id)
            logger.info("Oturum sonlandırıldı: %s", session_id)
            break

        if not user_input:
            continue

        # Çıkış komutu
        if user_input.lower() in {"exit", "quit"}:
            print("\n👋  İyi günler dileriz! / Goodbye!\n")
            save_conversation(messages, session_id)
            logger.info("Oturum sonlandırıldı: %s", session_id)
            break

        # Rezervasyon listesini göster
        if user_input.lower() == "rezervasyonlar":
            if os.path.exists(RESERVATIONS_FILE):
                with open(RESERVATIONS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"\n📋 Toplam {len(data)} rezervasyon bulundu:\n")
                for r in data:
                    print(f"  [{r['date']}] {r['details'][:100]}...")
                print()
            else:
                print("\n📋 Henüz kayıtlı rezervasyon yok.\n")
            continue

        messages.append({"role": "user", "content": user_input})
        logger.debug("Misafir: %s", user_input)

        assistant_reply = stream_response(client, messages)

        if assistant_reply:
            messages.append({"role": "assistant", "content": assistant_reply})
            logger.debug("Resepsiyon: %s", assistant_reply.strip())

            # Rezervasyon içeriyorsa kaydet
            rezervasyon_kelimeleri = [
                "rezervasyon", "reservation", "confirmed", "onaylandı",
                "booking", "kayıt", "reservierung"
            ]
            if any(k in assistant_reply.lower() for k in rezervasyon_kelimeleri):
                save_reservation(assistant_reply, session_id)


if __name__ == "__main__":
    main()
