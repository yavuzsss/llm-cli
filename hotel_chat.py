#!/usr/bin/env python3
# Renata Suites Boutique Hotel resepsiyon chatbotu — OpenAI Agents SDK versiyonu
# Önceki versiyona göre run_agent, execute_tool, regex hata yakalama tamamen kaldırıldı
# Bunların hepsi SDK tarafından otomatik olarak halloluyor

import os
import sys
import json
import logging
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, set_default_openai_client, set_default_openai_api, ModelSettings
from agents.exceptions import MaxTurnsExceeded

load_dotenv()

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------

LOG_FILE = "hotel_chat.log"
RESERVATIONS_FILE = "reservations.json"
ARCHIVE_FILE = "reservations_archive.json"

MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("MODEL_API_BASE_URL", "https://api.groq.com/openai/v1")

ROOM_CAPACITY = {
    "standart": 10,
    "deluxe": 12,
    "suite": 8,
    "apart": 7
}

ROOM_PRICES = {
    "standart": 4500,
    "deluxe": 5500,
    "suite": 7500,
    "apart": 9000
}

# ---------------------------------------------------------------------------
# Groq bağlantısını SDK'ya tanıt
# ---------------------------------------------------------------------------

groq_client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
set_default_openai_client(groq_client)
set_default_openai_api("chat_completions")  # Groq, Responses API değil Chat Completions kullanıyor

# Groq kullandığımız için OpenAI'ye tracing göndermeye gerek yok
from agents import set_tracing_disabled
set_tracing_disabled(True)

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
# Rezervasyon dosyası yardımcıları
# ---------------------------------------------------------------------------

def load_reservations() -> list:
    """reservations.json dosyasından geçerli rezervasyonları okur."""
    if not os.path.exists(RESERVATIONS_FILE):
        return []
    try:
        with open(RESERVATIONS_FILE, "r", encoding="utf-8") as f:
            all_reservations = json.load(f)
        required_fields = {"guest_name", "room_type", "checkin_date", "checkout_date"}
        return [r for r in all_reservations if required_fields.issubset(r.keys())]
    except Exception:
        return []


def save_reservations(reservations: list):
    with open(RESERVATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(reservations, f, ensure_ascii=False, indent=2)


def archive_past_reservations():
    """Çıkış tarihi geçmiş rezervasyonları arşive taşır."""
    reservations = load_reservations()
    today = date.today()
    active, archived = [], []

    for r in reservations:
        try:
            checkout = datetime.strptime(r["checkout_date"], "%Y-%m-%d").date()
            (archived if checkout < today else active).append(r)
        except (KeyError, ValueError):
            active.append(r)

    if not archived:
        return

    existing_archive = []
    if os.path.exists(ARCHIVE_FILE):
        try:
            with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                existing_archive = json.load(f)
        except Exception:
            pass

    existing_archive.extend(archived)
    with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_archive, f, ensure_ascii=False, indent=2)

    save_reservations(active)
    logger.info("%d rezervasyon arşivlendi.", len(archived))
    print(f"\n📦  {len(archived)} eski rezervasyon arşivlendi.\n")

# ---------------------------------------------------------------------------
# Tool'lar — @function_tool dekoratörü ile tanımlıyoruz
# SDK bunları otomatik olarak JSON schema'ya çeviriyor
# ---------------------------------------------------------------------------

@function_tool
def check_availability(checkin_date: str, checkout_date: str, room_type: str = "") -> str:
    """
    Belirli tarihler için oda müsaitliğini kontrol eder.
    checkin_date ve checkout_date YYYY-MM-DD formatında olmalı.
    room_type: standart, deluxe, suite veya apart. Boş bırakılırsa tüm odalar kontrol edilir.
    """
    if room_type == "":
        room_type = None

    try:
        checkin = datetime.strptime(checkin_date, "%Y-%m-%d").date()
        checkout = datetime.strptime(checkout_date, "%Y-%m-%d").date()
    except ValueError:
        return json.dumps({"error": "Tarih formatı hatalı. YYYY-MM-DD formatında olmalı."})

    if checkin >= checkout:
        return json.dumps({"error": "Çıkış tarihi giriş tarihinden sonra olmalıdır."})

    if checkin < date.today():
        return json.dumps({"error": "Geçmiş bir tarih için rezervasyon yapılamaz."})

    reservations = load_reservations()
    booked_counts = {}
    current = checkin
    while current < checkout:
        date_str = current.strftime("%Y-%m-%d")
        booked_counts[date_str] = {"standart": 0, "deluxe": 0, "suite": 0, "apart": 0}
        current += timedelta(days=1)

    for res in reservations:
        try:
            res_checkin = datetime.strptime(res["checkin_date"], "%Y-%m-%d").date()
            res_checkout = datetime.strptime(res["checkout_date"], "%Y-%m-%d").date()
            res_room = res["room_type"].lower()
            if res_checkin < checkout and res_checkout > checkin:
                cur = max(checkin, res_checkin)
                end = min(checkout, res_checkout)
                while cur < end:
                    date_str = cur.strftime("%Y-%m-%d")
                    if date_str in booked_counts and res_room in booked_counts[date_str]:
                        booked_counts[date_str][res_room] += 1
                    cur += timedelta(days=1)
        except (KeyError, ValueError):
            continue

    availability = {}
    for room, capacity in ROOM_CAPACITY.items():
        if room_type and room_type.lower() != room:
            continue
        min_available = capacity
        for date_str in booked_counts:
            available = capacity - booked_counts[date_str].get(room, 0)
            min_available = min(min_available, available)
        availability[room] = {
            "capacity": capacity,
            "available": min_available,
            "price_per_night": ROOM_PRICES[room]
        }

    return json.dumps({
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "nights": (checkout - checkin).days,
        "availability": availability
    }, ensure_ascii=False)


@function_tool
def make_reservation(guest_name: str, room_type: str, checkin_date: str,
                     checkout_date: str, num_guests: int) -> str:
    """
    Rezervasyon oluşturur ve reservations.json dosyasına kaydeder.
    room_type: standart, deluxe, suite veya apart
    Tarihler YYYY-MM-DD formatında olmalı.
    """
    avail_result = _check_availability_internal(checkin_date, checkout_date, room_type)

    if "error" in avail_result:
        return json.dumps({"success": False, "message": avail_result["error"]})

    room_key = room_type.lower()
    if room_key not in avail_result["availability"]:
        return json.dumps({"success": False, "message": f"Geçersiz oda tipi: {room_type}"})

    if avail_result["availability"][room_key]["available"] <= 0:
        return json.dumps({"success": False, "message": f"{checkin_date} - {checkout_date} tarihleri için {room_type} oda müsait değil."})

    nights = avail_result["nights"]
    price_per_night = ROOM_PRICES.get(room_key, 0)

    reservation = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "guest_name": guest_name,
        "room_type": room_key,
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "num_guests": num_guests,
        "nights": nights,
        "price_per_night": price_per_night,
        "total_price": price_per_night * nights,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    reservations = load_reservations()
    reservations.append(reservation)
    save_reservations(reservations)
    logger.info("Rezervasyon oluşturuldu: %s - %s - %s", guest_name, room_type, checkin_date)

    return json.dumps({
        "success": True,
        "reservation_id": reservation["id"],
        "guest_name": guest_name,
        "room_type": room_key,
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "nights": nights,
        "price_per_night": price_per_night,
        "total_price": reservation["total_price"],
        "message": "Rezervasyon başarıyla oluşturuldu!"
    }, ensure_ascii=False)


@function_tool
def get_reservations(guest_name: str = "") -> str:
    """
    Kayıtlı rezervasyonları getirir.
    guest_name verilirse sadece o misafirin rezervasyonları döner.
    Misafir rezervasyonunu sorduğunda veya adını söylediğinde çağır.
    """
    reservations = load_reservations()
    if guest_name:
        reservations = [r for r in reservations if guest_name.lower() in r.get("guest_name", "").lower()]
    return json.dumps({"reservations": reservations, "total": len(reservations)}, ensure_ascii=False)


@function_tool
def extend_reservation(reservation_id: str, new_checkout_date: str) -> str:
    """
    Mevcut bir rezervasyonun çıkış tarihini uzatır.
    Önce get_reservations ile rezervasyon ID'sini bul, sonra bu tool'u çağır.
    """
    reservations = load_reservations()
    target = next((r for r in reservations if r.get("id") == reservation_id), None)

    if not target:
        return json.dumps({"success": False, "message": f"Rezervasyon bulunamadı: {reservation_id}"})

    try:
        new_checkout = datetime.strptime(new_checkout_date, "%Y-%m-%d").date()
        old_checkout = datetime.strptime(target["checkout_date"], "%Y-%m-%d").date()
        checkin = datetime.strptime(target["checkin_date"], "%Y-%m-%d").date()
    except ValueError:
        return json.dumps({"success": False, "message": "Tarih formatı hatalı."})

    if new_checkout <= old_checkout:
        return json.dumps({"success": False, "message": "Yeni çıkış tarihi mevcut tarihten sonra olmalıdır."})

    avail_result = _check_availability_internal(old_checkout.strftime("%Y-%m-%d"), new_checkout_date, target["room_type"])
    if "error" in avail_result:
        return json.dumps({"success": False, "message": avail_result["error"]})

    if avail_result["availability"].get(target["room_type"], {}).get("available", 0) <= 0:
        return json.dumps({"success": False, "message": "Bu tarihler için oda müsait değil."})

    new_nights = (new_checkout - checkin).days
    target["checkout_date"] = new_checkout_date
    target["nights"] = new_nights
    target["total_price"] = new_nights * target["price_per_night"]
    save_reservations(reservations)
    logger.info("Rezervasyon uzatıldı: %s → %s", reservation_id, new_checkout_date)

    return json.dumps({
        "success": True,
        "message": "Rezervasyon başarıyla uzatıldı.",
        "reservation_id": reservation_id,
        "new_checkout_date": new_checkout_date,
        "total_nights": new_nights,
        "total_price": target["total_price"]
    }, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Otel bilgileri ve sistem prompt
# ---------------------------------------------------------------------------

def _check_availability_internal(checkin_date: str, checkout_date: str, room_type: str = None) -> dict:
    """check_availability'nin iç versiyonu — diğer tool'lar tarafından çağrılır."""
    if room_type == "" or room_type is None:
        room_type = None

    try:
        checkin = datetime.strptime(checkin_date, "%Y-%m-%d").date()
        checkout = datetime.strptime(checkout_date, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "Tarih formatı hatalı."}

    if checkin >= checkout:
        return {"error": "Çıkış tarihi giriş tarihinden sonra olmalıdır."}

    if checkin < date.today():
        return {"error": "Geçmiş bir tarih için rezervasyon yapılamaz."}

    reservations = load_reservations()
    booked_counts = {}
    current = checkin
    while current < checkout:
        date_str = current.strftime("%Y-%m-%d")
        booked_counts[date_str] = {"standart": 0, "deluxe": 0, "suite": 0, "apart": 0}
        current += timedelta(days=1)

    for res in reservations:
        try:
            res_checkin = datetime.strptime(res["checkin_date"], "%Y-%m-%d").date()
            res_checkout = datetime.strptime(res["checkout_date"], "%Y-%m-%d").date()
            res_room = res["room_type"].lower()
            if res_checkin < checkout and res_checkout > checkin:
                cur = max(checkin, res_checkin)
                end = min(checkout, res_checkout)
                while cur < end:
                    date_str = cur.strftime("%Y-%m-%d")
                    if date_str in booked_counts and res_room in booked_counts[date_str]:
                        booked_counts[date_str][res_room] += 1
                    cur += timedelta(days=1)
        except (KeyError, ValueError):
            continue

    availability = {}
    for room, capacity in ROOM_CAPACITY.items():
        if room_type and room_type.lower() != room:
            continue
        min_available = capacity
        for date_str in booked_counts:
            available = capacity - booked_counts[date_str].get(room, 0)
            min_available = min(min_available, available)
        availability[room] = {
            "capacity": capacity,
            "available": min_available,
            "price_per_night": ROOM_PRICES[room]
        }

    return {
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "nights": (checkout - checkin).days,
        "availability": availability
    }

HOTEL_DATA = """
OTEL BİLGİLERİ:
- Otel Adı: Renata Suites Boutique Hotel
- Adres: Nakiye Elgün Sk. No:44, Osmanbey/Şişli, İstanbul
- Yıldız: 4 yıldızlı butik otel
- Toplam Oda Sayısı: 37

CHECK-IN / CHECK-OUT:
- Check-in: 14:00
- Check-out: 12:00
- Erken check-in: Talep üzerine, müsaitliğe göre (ek ücret uygulanabilir)
- Geç check-out: Talep üzerine, müsaitliğe göre (ek ücret uygulanabilir)
- Temassız (contactless) check-in/check-out imkânı mevcuttur

ODA TİPLERİ VE GECELİK FİYATLAR:
- Standart Oda: 4500 TL — Klimalı, 46 inç Smart TV, minibar, özel banyo, bornoz ve terlik (10 oda)
- Deluxe Oda: 5500 TL — Geniş oturma alanı, şehir manzarası, çalışma masası (12 oda)
- Suite: 7500 TL — 40 m², iş seyahati için ideal, çalışma masası (8 oda)
- Apart Suite: 9000 TL — 45-53 m², uzun konaklamalar için, mutfak ve çalışma alanı (7 oda)

HİZMETLER:
- Ücretsiz Wi-Fi (tüm alanlarda)
- Açık büfe kahvaltı — hafta içi 07:30-10:30, hafta sonu 07:30-11:00 (oda fiyatına dahil)
- Helal ve glütensiz kahvaltı seçenekleri
- Restoran, 2 Bar/Lounge
- Spa, sauna ve buhar odası (ek ücretli)
- Fitness merkezi, Oda servisi (24 saat)
- Vale otopark, Havalimanı transferi (ek ücretli)
- Kuru temizleme, Toplantı salonu, Concierge (7/24)

KONUM:
- Taksim Meydanı: 4 dakika (metro ile)
- Osmanbey Metro: Yürüme mesafesinde
- İstanbul Havalimanı: 47 km

ÖDEME: Visa, Mastercard, Amex, Nakit
DEPOZITO: Check-in sırasında kredi kartından provizyon alınır

İLETİŞİM:
- Telefon: +90 212 282 42 42
- E-posta: info@renatahotel.com
- Web: www.renatahotel.com
"""


def build_instructions() -> str:
    """Her oturum başında güncel tarihle sistem talimatları oluşturur."""
    now = datetime.now()
    tomorrow = now + timedelta(days=1)
    return f"""Sen Renata Suites Boutique Hotel'in profesyonel ve güler yüzlü resepsiyon asistanısın.

BUGÜNÜN TARİHİ: {now.strftime("%d %B %Y")} — Saat: {now.strftime("%H:%M")}
YARIN: {tomorrow.strftime("%d %B %Y")}
"Yarın", "bu hafta sonu" gibi ifadeleri yukarıdaki tarihe göre YYYY-MM-DD formatına çevir.

DİL KURALI: Misafirin yazdığı dilde cevap ver. Asla dil karıştırma.
- Türkçe mesaj → Türkçe cevap
- İngilizce mesaj → İngilizce cevap
- Almanca mesaj → Almanca cevap

MÜSAİTLİK: Oda sorulduğunda check_availability çağır. Selamlama ve genel sorularda tool çağırma.

REZERVASYON KURALI (ÇOK ÖNEMLİ):
- make_reservation çağırmadan önce şu 5 bilgiyi MUTLAKA topla:
  1. Ad Soyad — misafir söylemeden asla varsayma, sor
  2. Oda tipi (standart / deluxe / suite / apart)
  3. Giriş tarihi
  4. Çıkış tarihi  
  5. Misafir sayısı
- Eksik bilgi varsa make_reservation ÇAĞIRMA, önce sor
- guest_name alanına asla "misafir", "guest" veya boş değer yazma

SORGULAMA: Misafir adını söylediğinde veya rezervasyonunu sorduğunda get_reservations çağır.
Çağırmadan "rezervasyon bulunamıyor" deme.

UZATMA: Önce get_reservations ile ID bul, sonra extend_reservation çağır.

FİYAT: Toplam = gecelik fiyat x gece sayısı. Hesabı açıkça göster.

GENEL: Kısa ve samimi cevaplar ver. Sadece otel konularında yardımcı ol.

{HOTEL_DATA}"""

# ---------------------------------------------------------------------------
# Dil tespiti
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Unicode ve kelime analizi ile dil tespiti yapar. API çağrısı yapmaz."""
    if any("\u0600" <= c <= "\u06ff" for c in text):
        return "Arapça"

    if any(c in "çğışöüÇĞİŞÖÜ" for c in text):
        return "Türkçe"

    if any(c in "äöüßÄÖÜ" for c in text):
        return "Almanca"

    if any(c in "àâæéèêëîïôœùûüÿÀÂÆÉÈÊËÎÏÔŒÙÛÜŸ" for c in text):
        return "Fransızca"

    english = {"hello","hi","hey","good","please","thank","thanks","yes","no",
               "what","how","i","we","need","want","room","book","can","the","a"}
    german = {"guten","hallo","bitte","danke","ja","nein","ich","bin","sie","und","nicht"}
    french = {"bonjour","bonsoir","merci","oui","non","je","vous","nous","avec","pour"}
    turkish = {"merhaba","selam","evet","hayır","lütfen","nasıl","iyi","tamam",
               "rezervasyon","oda","yarın","bugün","istiyorum","var","yok"}

    words = set(text.lower().split())
    scores = {
        "Türkçe": len(words & turkish),
        "İngilizce": len(words & english),
        "Almanca": len(words & german),
        "Fransızca": len(words & french),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Türkçe"

# ---------------------------------------------------------------------------
# Terminal yardımcıları
# ---------------------------------------------------------------------------

def show_reservations():
    reservations = load_reservations()
    if not reservations:
        print("\n📋 Henüz aktif rezervasyon yok.\n")
        return
    print(f"\n📋 Toplam {len(reservations)} aktif rezervasyon:\n")
    for r in reservations:
        print(f"  [{r.get('created_at', '?')}]")
        print(f"  Misafir : {r.get('guest_name', '?')}")
        print(f"  Oda     : {r.get('room_type', '?')}")
        print(f"  Giriş   : {r.get('checkin_date', '?')}")
        print(f"  Çıkış   : {r.get('checkout_date', '?')}")
        print(f"  Toplam  : {r.get('total_price', '?')} TL")
        print()


def print_banner(session_id: str):
    reservations = load_reservations()
    print("=" * 55)
    print("  🏨  Renata Suites Boutique Hotel — Resepsiyon")
    print(f"  Model      : {MODEL}")
    print(f"  API        : {API_BASE_URL}")
    print(f"  Oturum     : {session_id}")
    print(f"  Rezervasyon: {len(reservations)} aktif kayıt")
    print("  Çıkmak için 'exit' veya 'quit' yazın.")
    print("  Rezervasyon listesi için 'rezervasyonlar' yazın.")
    print("=" * 55)
    print("\nHoş geldiniz! / Welcome! / Willkommen!\n")

# ---------------------------------------------------------------------------
# Ana döngü
# ---------------------------------------------------------------------------

def main():
    if not API_KEY:
        print("\n❌  API anahtarı bulunamadı. Lütfen .env dosyanızı kontrol edin.\n")
        sys.exit(1)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Geçmiş rezervasyonları arşivle
    archive_past_reservations()

    print_banner(session_id)

    # Agent'ı oluştur — tool'lar ve talimatlarla birlikte
    agent = Agent(
        name="Renata Resepsiyon",
        model=MODEL,
        instructions=build_instructions(),
        tools=[check_availability, make_reservation, get_reservations, extend_reservation],
        model_settings=ModelSettings(parallel_tool_calls=False)
    )

    logger.info("Oturum başladı. ID: %s | Model: %s", session_id, MODEL)

    # Konuşma geçmişini tutan liste — SDK bunu otomatik yönetiyor
    conversation_history = []

    while True:
        try:
            user_input = input("💬 Misafir: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋  İyi günler dileriz! / Goodbye!\n")
            logger.info("Oturum sonlandırıldı: %s", session_id)
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("\n👋  İyi günler dileriz! / Goodbye!\n")
            logger.info("Oturum sonlandırıldı: %s", session_id)
            break

        if user_input.lower() == "rezervasyonlar":
            show_reservations()
            continue

        # Dil tespiti yapıp mesaja ekle
        detected_lang = detect_language(user_input)
        enriched_input = f"[Misafirin dili: {detected_lang}] {user_input}"
        logger.debug("Misafir (%s): %s", detected_lang, user_input)

        # Konuşma geçmişine ekle
        conversation_history.append({"role": "user", "content": enriched_input})

        try:
            # SDK agent'ı çalıştırır, tool call'ları otomatik halleder
            result = Runner.run_sync(
                agent,
                conversation_history,
                max_turns=10
            )

            reply = result.final_output
            print(f"\n🏨 Resepsiyon: {reply}\n")
            logger.debug("Resepsiyon: %s", reply)

            # Asistanın cevabını geçmişe ekle
            conversation_history.append({"role": "assistant", "content": reply})

        except MaxTurnsExceeded:
            print("\n⚠️  Üzgünüm, isteğinizi işleyemedim. Lütfen tekrar deneyin.\n")
            logger.warning("MaxTurnsExceeded")
        except Exception as e:
            print(f"\n❌  Bir hata oluştu: {e}\n")
            logger.error("Hata: %s", e)


if __name__ == "__main__":
    main()
