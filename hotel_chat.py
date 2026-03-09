#!/usr/bin/env python3
# Grand Yavuz Hotel için yaptığım resepsiyon chatbotu
# Groq API kullanıyor, çok dil destekliyor ve rezervasyon kaydediyor

import os
import sys
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI, APIError, AuthenticationError, RateLimitError

# .env dosyasındaki değişkenleri yükle (API anahtarı vs.)
load_dotenv()

# Otele ait tüm bilgileri buraya yazdım
# Asistan sadece bu bilgilere göre cevap veriyor, bir şey uydurmaya çalışmıyor
HOTEL_DATA = """
OTEL BİLGİLERİ:
- Otel Adı: Grand Yavuz Hotel
- Adres: Bağdat Caddesi No:42, İstanbul

CHECK-IN / CHECK-OUT:
- Check-in: 14:00
- Check-out: 12:00
- Erken check-in: Talep üzerine, müsaitliğe göre (ek ücret uygulanabilir)
- Geç check-out: Talep üzerine, müsaitliğe göre (ek ücret uygulanabilir)

ODA TİPLERİ VE GECELİK FİYATLAR:
- Standart Oda: 2500 TL — Çift kişilik yatak, şehir manzarası, 25 m²
- Deluxe Oda: 3500 TL — King yatak, deniz manzarası, 35 m²
- Suite: 6000 TL — King yatak + oturma odası, panoramik manzara, 60 m²
- Aile Odası: 4500 TL — 2 ayrı yatak, geniş banyo, 45 m²

HİZMETLER:
- Ücretsiz Wi-Fi (tüm alanlarda)
- Açık büfe kahvaltı (07:00 - 10:30) — oda fiyatına dahil
- Restoran (12:00 - 22:00)
- Spa ve wellness (09:00 - 21:00)
- Fitness salonu (24 saat)
- Açık yüzme havuzu (08:00 - 20:00, Mayıs-Ekim arası)
- Otopark (ücretli, 150 TL/gün)
- Havalimanı transferi (rezervasyon gerekli)
- Oda servisi (24 saat)
- Kuru temizleme (09:00 - 18:00)
- Concierge (24 saat)

ÖDEME:
- Kabul edilen kartlar: Visa, Mastercard, American Express
- Nakit ödeme kabul edilir
- Depozito: Check-in sırasında 500 TL

İPTAL POLİTİKASI:
- 48 saat öncesine kadar ücretsiz iptal
- 48 saat içinde iptal: 1 gecelik ücret tahsil edilir

İLETİŞİM:
- Telefon: +90 212 555 00 42
- E-posta: info@grandyavuzhotel.com
- Resepsiyon: 7/24
"""

# Asistana verdiğim talimatlar
# Misafirin dilini algılayıp o dilde cevap vermesini istedim
# Rezervasyon alırken sırayla bilgileri toplamasını söyledim
SYSTEM_PROMPT = f"""Sen Grand Yavuz Hotel'in resepsiyon asistanısın. Kibarlığın ve yardımseverliğinle tanınıyorsun.

DİL KURALI: Misafirin hangi dilde yazdığını anlayıp aynı dilde cevap ver.
Desteklenen diller: Türkçe, İngilizce, Almanca, Fransızca, Arapça.

REZERVASYON KURALI: Misafir rezervasyon yapmak istediğinde şu bilgileri sırayla iste:
1. Ad Soyad
2. Oda tipi (Standart / Deluxe / Suite / Aile)
3. Giriş tarihi
4. Çıkış tarihi
5. Misafir sayısı
Bilgileri aldıktan sonra rezervasyonu onayla ve toplam ücreti hesapla.

FİYAT HESAPLAMA: Toplam = gecelik fiyat x gece sayısı
Hesaplamayı misafire açıkça göster.

GENEL KURALLAR:
- Kısa ve samimi cevaplar ver
- Sadece otel konularında yardımcı ol
- Bilmediğin şeyleri uydurma, resepsiyona yönlendir

OTEL BİLGİLERİ:
{HOTEL_DATA}"""

# Dosya adları
LOG_FILE = "hotel_chat.log"
RESERVATIONS_FILE = "reservations.json"
HISTORY_FILE = "conversation_history.json"

# .env'den ayarları oku, yoksa varsayılanları kullan
MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("MODEL_API_BASE_URL", "https://api.groq.com/openai/v1")


# Log ayarları — hem dosyaya hem terminale yazıyor
# Dosyaya her şeyi yazıyor, terminale sadece hataları gösteriyor
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


# Oturum bitince tüm konuşmayı JSON dosyasına kaydediyorum
# System prompt'u kaydetmiyorum çünkü zaten kodda var, gereksiz yer kaplar
def save_conversation(messages: list[dict], session_id: str):
    history = []

    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    conversation = {
        "session_id": session_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": [m for m in messages if m["role"] != "system"]
    }

    history.append(conversation)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    logger.info("Konuşma geçmişi kaydedildi: %s", HISTORY_FILE)


# Asistan rezervasyon onayladığında bunu ayrı bir dosyaya da kaydediyorum
# Böylece tüm rezervasyonları tek yerden görebiliyorum
def save_reservation(reservation_text: str, session_id: str):
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


# API bağlantısını kuruyorum
# API anahtarı yoksa program çalışmaz, kullanıcıya açıklıyorum
def get_client() -> OpenAI:
    if not API_KEY:
        logger.error("OPENAI_API_KEY bulunamadı.")
        print("\n❌  API anahtarı bulunamadı. Lütfen .env dosyanızı kontrol edin.\n")
        sys.exit(1)
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


# Cevabı streaming ile alıyorum — ChatGPT gibi kelime kelime ekrana yazıyor
# Tüm cevap gelene kadar beklemek yerine anlık göstermek daha iyi hissettiriyor
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


# Program açılınca gösterilen karşılama ekranı
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


def main():
    # Her oturuma tarih/saat bazlı benzersiz bir ID veriyorum
    # Böylece hangi konuşmanın hangi oturuma ait olduğunu anlayabiliyorum
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print_banner(session_id)

    client = get_client()

    # Konuşma geçmişi burada tutuluyor
    # Her mesaj listeye ekleniyor, asistan önceki mesajları hatırlıyor
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

        if user_input.lower() in {"exit", "quit"}:
            print("\n👋  İyi günler dileriz! / Goodbye!\n")
            save_conversation(messages, session_id)
            logger.info("Oturum sonlandırıldı: %s", session_id)
            break

        # Kullanıcı "rezervasyonlar" yazarsa kayıtlı rezervasyonları göster
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

            # Asistanın cevabında rezervasyon ile ilgili anahtar kelimeler varsa kaydet
            rezervasyon_kelimeleri = [
                "rezervasyon", "reservation", "confirmed", "onaylandı",
                "booking", "kayıt", "reservierung"
            ]
            if any(k in assistant_reply.lower() for k in rezervasyon_kelimeleri):
                save_reservation(assistant_reply, session_id)


if __name__ == "__main__":
    main()
