#!/usr/bin/env python3
"""
Otel Resepsiyon Asistanı — CLI LLM Uygulaması
Kullanıcıların otel hakkında sorularını yanıtlayan bir CLI chatbot.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from openai import OpenAI, APIError, AuthenticationError, RateLimitError

# ---------------------------------------------------------------------------
# Konfigürasyon
# ---------------------------------------------------------------------------

load_dotenv()

HOTEL_DATA = """
OTEL BİLGİLERİ:
- Otel Adı: Grand Yavuz Hotel
- Adres: Bağdat Caddesi No:42, İstanbul

CHECK-IN / CHECK-OUT:
- Check-in saati: 14:00
- Check-out saati: 12:00
- Erken check-in: Talep üzerine, müsaitliğe göre (ek ücret uygulanabilir)
- Geç check-out: Talep üzerine, müsaitliğe göre (ek ücret uygulanabilir)

ODA TİPLERİ:
- Standart Oda: 1 çift kişilik yatak, şehir manzarası, 25 m²
- Deluxe Oda: 1 king yatak, deniz manzarası, 35 m²
- Suite: 1 king yatak + oturma odası, panoramik manzara, 60 m²
- Aile Odası: 2 ayrı yatak, geniş banyo, 45 m²

HİZMETLER:
- Ücretsiz Wi-Fi (tüm alanlarda)
- Açık büfe kahvaltı (07:00 - 10:30)
- Restoran (12:00 - 22:00)
- Spa ve wellness merkezi (09:00 - 21:00)
- Fitness salonu (24 saat)
- Açık yüzme havuzu (08:00 - 20:00, Mayıs-Ekim arası)
- Otopark (ücretli, günlük 150 TL)
- Havalimanı transfer (rezervasyon gerekli)
- Oda servisi (24 saat)
- Kuru temizleme (09:00 - 18:00)
- Concierge hizmeti (24 saat)

ÖDEME:
- Kabul edilen kartlar: Visa, Mastercard, American Express
- Nakit ödeme kabul edilmektedir
- Depozito: Check-in sırasında 500 TL

İPTAL POLİTİKASI:
- 48 saat öncesine kadar ücretsiz iptal
- 48 saat içinde iptal: 1 gecelik ücret tahsil edilir

İLETİŞİM:
- Telefon: +90 212 555 00 42
- E-posta: info@grandyavuzhotel.com
- Resepsiyon: 7/24 hizmetinizdedir
"""

SYSTEM_PROMPT = f"""Sen Grand Yavuz Hotel'in profesyonel ve güler yüzlü resepsiyon asistanısın.
Misafirlere Türkçe olarak yardımcı oluyorsun.
Cevapların kısa, net ve samimi olmalı.
Sadece otel ile ilgili konularda yardımcı oluyorsun.
Bilmediğin bir şey sorulursa, misafiri resepsiyona yönlendir.

Aşağıda otele ait bilgiler bulunmaktadır, yalnızca bu bilgilere dayanarak cevap ver:

{HOTEL_DATA}"""

LOG_FILE = "hotel_chat.log"
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


def print_banner():
    print("=" * 55)
    print("  🏨  Grand Yavuz Hotel — Resepsiyon Asistanı")
    print(f"  Model : {MODEL}")
    print(f"  API   : {API_BASE_URL}")
    print("  Çıkmak için 'exit' veya 'quit' yazın.")
    print("=" * 55)
    print("\nHoş geldiniz! Size nasıl yardımcı olabilirim?\n")

# ---------------------------------------------------------------------------
# Ana Döngü
# ---------------------------------------------------------------------------

def main():
    print_banner()

    client = get_client()
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    logger.info("Oturum başladı. Model: %s | API: %s", MODEL, API_BASE_URL)

    while True:
        try:
            user_input = input("💬 Misafir: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋  İyi günler dileriz!\n")
            logger.info("Oturum sonlandırıldı.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("\n👋  İyi günler dileriz!\n")
            logger.info("Oturum sonlandırıldı.")
            break

        messages.append({"role": "user", "content": user_input})
        logger.debug("Misafir: %s", user_input)

        assistant_reply = stream_response(client, messages)

        if assistant_reply:
            messages.append({"role": "assistant", "content": assistant_reply})
            logger.debug("Resepsiyon: %s", assistant_reply.strip())


if __name__ == "__main__":
    main()
