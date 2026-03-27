#!/usr/bin/env python3
# Renata Suites Boutique Hotel resepsiyon chatbotu — gelişmiş versiyon
# Tool calling ve agent döngüsü ile rezervasyon yönetimi yapıyor

import os
import sys
import json
import logging
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import re
from openai import OpenAI, APIError, AuthenticationError, RateLimitError

load_dotenv()

# Otele ait bilgiler — asistan sadece bunlara dayanarak cevap veriyor
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
- Suite: 7500 TL — 40 m², iş seyahati için ideal, çalışma masası, ücretsiz Wi-Fi (8 oda)
- Apart Suite: 9000 TL — 45-53 m², uzun konaklamalar için, mutfak ve çalışma alanı (7 oda)

NOT: Fiyatlar değişkenlik gösterebilir. Güncel fiyatlar için resepsiyonumuzu arayınız.

HİZMETLER:
- Ücretsiz Wi-Fi (tüm alanlarda)
- Açık büfe kahvaltı — hafta içi 07:30-10:30, hafta sonu 07:30-11:00 (oda fiyatına dahil)
- Helal ve glütensiz kahvaltı seçenekleri mevcuttur
- Restoran (öğle ve akşam yemeği)
- 2 Bar/Lounge
- Spa, sauna ve buhar odası (ek ücretli)
- Fitness merkezi
- Oda servisi (24 saat)
- Vale otopark
- Havalimanı transferi (ek ücretli, rezervasyon gerekli)
- Kuru temizleme ve çamaşırhane hizmeti
- Toplantı salonu ve iş merkezi
- Concierge (7/24)
- Tur masası
- Engelli odası mevcuttur
- Çevre dostu otel (ozon teknolojisi ile oda temizliği)

KONUM:
- Taksim Meydanı: 4 dakika (metro ile)
- Osmanbey Metro İstasyonu: Yürüme mesafesinde
- Cevahir AVM: 1.8 km
- İstanbul Kongre Merkezi: 900 metre
- İstanbul Havalimanı: 47 km

ÖDEME:
- Kabul edilen kartlar: Visa, Mastercard, American Express
- Nakit ödeme kabul edilir
- Depozito: Check-in sırasında kredi kartından provizyon alınır

İPTAL POLİTİKASI:
- İptal koşulları oda tipine ve rezervasyon tarihine göre değişmektedir
- Detaylı bilgi için resepsiyonumuzu arayınız

İLETİŞİM:
- Telefon: +90 212 282 42 42
- E-posta: info@renatahotel.com
- Web: www.renatahotel.com
- Resepsiyon: 7/24
"""

# Asistana verilen talimatlar — tarih programa her başlangıçta otomatik ekleniyor
# Böylece asistan "yarın", "bu hafta sonu" gibi ifadeleri doğru hesaplayabiliyor
def build_system_prompt() -> str:
    now = datetime.now()
    tomorrow = now + timedelta(days=1)
    return f"""Sen Renata Suites Boutique Hotel'in profesyonel ve güler yüzlü resepsiyon asistanısın.

BUGÜNÜN TARİHİ: {now.strftime("%d %B %Y")} — Saat: {now.strftime("%H:%M")}
YARIN: {tomorrow.strftime("%d %B %Y")}
Tarihleri hesaplarken yukarıdaki bugünün tarihini kullan. "Yarın", "bu hafta sonu" gibi ifadeleri buna göre YYYY-MM-DD formatına çevir.

DİL KURALI (ZORUNLU): Her mesajın başında [Misafirin dili: X] etiketi bulunur. Bu etiket sadece senin için bir talimattır, cevabında bu etiketi ASLA yazma.
- [Misafirin dili: İngilizce] → sadece İngilizce yaz
- [Misafirin dili: Almanca] → sadece Almanca yaz
- [Misafirin dili: Fransızca] → sadece Fransızca yaz
- [Misafirin dili: Türkçe] → sadece Türkçe yaz
- İki dili ASLA karıştırma, tek bir dilde cevap ver
- Cevabının başına [Misafirin dili: ...] YAZMA

MÜSAİTLİK KURALI: Sadece misafir açıkça oda müsaitliği sorduğunda veya rezervasyon yapmak istediğinde check_availability aracını çağır. Selamlama ve genel sorularda kesinlikle tool çağırma.

REZERVASYON KURALI: Misafir rezervasyon yapmak istediğinde şu bilgileri sırayla iste:
1. Ad Soyad
2. Oda tipi (Standart / Deluxe / Suite / Apart)
3. Giriş tarihi
4. Çıkış tarihi
5. Misafir sayısı
Tüm bilgileri aldıktan sonra make_reservation aracını çağır. Araç başarılı dönerse rezervasyonu onayla ve toplam ücreti göster.

REZERVASYONu SORGULAMA KURALI: Misafir adını söylediğinde veya "rezervasyonum var", "kaydım var" gibi bir şey söylediğinde MUTLAKA get_reservations aracını çağır ve sistemde ara. Aracı çağırmadan "rezervasyon bulunamıyor" deme. Araç sonucuna göre cevap ver.

REZERVASYONu UZATMA KURALI: Misafir rezervasyonunu uzatmak istediğinde:
1. Önce get_reservations aracını çağır ve misafirin rezervasyonunu bul
2. Kaç gün uzatmak istediğini öğren
3. Yeni çıkış tarihini hesapla
4. extend_reservation aracını çağır
5. Misafirden tekrar bilgi ISTEME, rezervasyon sistemde zaten kayıtlı

DİL TEMİZLİĞİ: Cevaplarında sadece misafirin dilini kullan. Rusça, Almanca veya başka dillerden kelime karıştırma.

FİYAT HESAPLAMA: Toplam = gecelik fiyat x gece sayısı. Hesaplamayı misafire açıkça göster.

GENEL KURALLAR:
- Kısa ve samimi cevaplar ver
- Sadece otel konularında yardımcı ol
- Bilmediğin şeyleri uydurma, resepsiyona yönlendir

OTEL BİLGİLERİ:
{HOTEL_DATA}"""

# Dosya adları
LOG_FILE = "hotel_chat.log"
RESERVATIONS_FILE = "reservations.json"
ARCHIVE_FILE = "reservations_archive.json"
HISTORY_FILE = "conversation_history.json"

# .env'den ayarları oku
MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("MODEL_API_BASE_URL", "https://api.groq.com/openai/v1")

# Her oda tipinde kaç oda olduğunu burada tutuyorum
ROOM_CAPACITY = {
    "standart": 10,
    "deluxe": 12,
    "suite": 8,
    "apart": 7
}

# Oda fiyatları
ROOM_PRICES = {
    "standart": 4500,
    "deluxe": 5500,
    "suite": 7500,
    "apart": 9000
}

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
# Tool fonksiyonları — asistanın çağırabileceği araçlar
# ---------------------------------------------------------------------------

def check_availability(checkin_date: str, checkout_date: str, room_type: str = None) -> dict:
    """Belirli tarihler için oda müsaitliğini kontrol eder."""
    # Boş string gelirse None olarak işle
    if room_type == "":
        room_type = None
    try:
        checkin = datetime.strptime(checkin_date, "%Y-%m-%d").date()
        checkout = datetime.strptime(checkout_date, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "Tarih formatı hatalı. YYYY-MM-DD formatında olmalı."}

    if checkin >= checkout:
        return {"error": "Çıkış tarihi giriş tarihinden sonra olmalıdır."}

    if checkin < date.today():
        return {"error": "Geçmiş bir tarih için rezervasyon yapılamaz."}

    # Mevcut rezervasyonları oku
    reservations = load_reservations()

    # Kontrol edilecek tarihler arasındaki her gün için dolu oda sayısını hesapla
    booked_counts = {}
    current = checkin
    while current < checkout:
        date_str = current.strftime("%Y-%m-%d")
        booked_counts[date_str] = {"standart": 0, "deluxe": 0, "suite": 0, "apart": 0}
        current += timedelta(days=1)

    # Her rezervasyonun bu tarih aralığıyla çakışıp çakışmadığını kontrol et
    for res in reservations:
        try:
            res_checkin = datetime.strptime(res["checkin_date"], "%Y-%m-%d").date()
            res_checkout = datetime.strptime(res["checkout_date"], "%Y-%m-%d").date()
            res_room = res["room_type"].lower()

            # Çakışma var mı?
            if res_checkin < checkout and res_checkout > checkin:
                current = max(checkin, res_checkin)
                end = min(checkout, res_checkout)
                while current < end:
                    date_str = current.strftime("%Y-%m-%d")
                    if date_str in booked_counts and res_room in booked_counts[date_str]:
                        booked_counts[date_str][res_room] += 1
                    current += timedelta(days=1)
        except (KeyError, ValueError):
            continue

    # Müsait oda sayısını hesapla
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

    nights = (checkout - checkin).days
    return {
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "nights": nights,
        "availability": availability
    }


def make_reservation(guest_name: str, room_type: str, checkin_date: str,
                     checkout_date: str, num_guests: int) -> dict:
    """Rezervasyon oluşturur ve reservations.json dosyasına kaydeder."""

    # Önce müsaitliği kontrol et
    availability_result = check_availability(checkin_date, checkout_date, room_type)

    if "error" in availability_result:
        return {"success": False, "message": availability_result["error"]}

    room_key = room_type.lower()
    if room_key not in availability_result["availability"]:
        return {"success": False, "message": f"Geçersiz oda tipi: {room_type}"}

    available = availability_result["availability"][room_key]["available"]
    if available <= 0:
        return {"success": False, "message": f"Üzgünüz, {checkin_date} - {checkout_date} tarihleri için {room_type} oda müsait değil."}

    # Rezervasyonu oluştur
    nights = availability_result["nights"]
    price_per_night = ROOM_PRICES.get(room_key, 0)
    total_price = price_per_night * nights

    reservation = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "guest_name": guest_name,
        "room_type": room_key,
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "num_guests": num_guests,
        "nights": nights,
        "price_per_night": price_per_night,
        "total_price": total_price,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Dosyaya kaydet
    reservations = load_reservations()
    reservations.append(reservation)
    save_reservations(reservations)

    logger.info("Rezervasyon oluşturuldu: %s - %s - %s", guest_name, room_type, checkin_date)

    return {
        "success": True,
        "reservation_id": reservation["id"],
        "guest_name": guest_name,
        "room_type": room_key,
        "checkin_date": checkin_date,
        "checkout_date": checkout_date,
        "nights": nights,
        "price_per_night": price_per_night,
        "total_price": total_price,
        "message": "Rezervasyon başarıyla oluşturuldu!"
    }


# ---------------------------------------------------------------------------
# Rezervasyon dosyası yardımcıları
# ---------------------------------------------------------------------------

def load_reservations() -> list:
    """reservations.json dosyasından rezervasyonları okur.
    Gerekli alanları olmayan eski format kayıtları otomatik olarak atlar."""
    if not os.path.exists(RESERVATIONS_FILE):
        return []
    try:
        with open(RESERVATIONS_FILE, "r", encoding="utf-8") as f:
            all_reservations = json.load(f)
        # Sadece gerekli alanları olan yeni format kayıtları al
        required_fields = {"guest_name", "room_type", "checkin_date", "checkout_date"}
        valid = [r for r in all_reservations if required_fields.issubset(r.keys())]
        return valid
    except Exception:
        return []


def save_reservations(reservations: list):
    """Rezervasyonları reservations.json dosyasına yazar."""
    with open(RESERVATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(reservations, f, ensure_ascii=False, indent=2)


def archive_past_reservations():
    """Çıkış tarihi geçmiş rezervasyonları arşive taşır."""
    reservations = load_reservations()
    today = date.today()

    active = []
    archived = []

    for r in reservations:
        try:
            checkout = datetime.strptime(r["checkout_date"], "%Y-%m-%d").date()
            if checkout < today:
                archived.append(r)
            else:
                active.append(r)
        except (KeyError, ValueError):
            active.append(r)

    if not archived:
        return

    # Arşiv dosyasına ekle
    existing_archive = []
    if os.path.exists(ARCHIVE_FILE):
        try:
            with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                existing_archive = json.load(f)
        except Exception:
            existing_archive = []

    existing_archive.extend(archived)

    with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_archive, f, ensure_ascii=False, indent=2)

    # Aktif rezervasyonları kaydet
    save_reservations(active)

    logger.info("%d rezervasyon arşivlendi.", len(archived))
    if archived:
        print(f"\n📦  {len(archived)} eski rezervasyon arşivlendi.\n")


# ---------------------------------------------------------------------------
# Tool tanımları — asistana hangi araçların olduğunu söylüyoruz
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Belirli tarihler için oda müsaitliğini kontrol eder. Misafir oda sorduğunda mutlaka çağır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "checkin_date": {
                        "type": "string",
                        "description": "Giriş tarihi YYYY-MM-DD formatında. Örnek: 2026-03-20"
                    },
                    "checkout_date": {
                        "type": "string",
                        "description": "Çıkış tarihi YYYY-MM-DD formatında. Örnek: 2026-03-22"
                    },
                    "room_type": {
                        "type": "string",
                        "description": "Oda tipi: standart, deluxe, suite veya apart. Belirtilmezse tüm odalar kontrol edilir. Boş bırakılabilir."
                    }
                },
                "required": ["checkin_date", "checkout_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "make_reservation",
            "description": "Rezervasyon oluşturur ve kalıcı olarak kaydeder. Tüm bilgiler toplandıktan sonra çağır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "guest_name": {
                        "type": "string",
                        "description": "Misafirin adı ve soyadı"
                    },
                    "room_type": {
                        "type": "string",
                        "description": "Oda tipi: standart, deluxe, suite veya apart",
                        "enum": ["standart", "deluxe", "suite", "apart"]
                    },
                    "checkin_date": {
                        "type": "string",
                        "description": "Giriş tarihi YYYY-MM-DD formatında"
                    },
                    "checkout_date": {
                        "type": "string",
                        "description": "Çıkış tarihi YYYY-MM-DD formatında"
                    },
                    "num_guests": {
                        "type": "integer",
                        "description": "Misafir sayısı"
                    }
                },
                "required": ["guest_name", "room_type", "checkin_date", "checkout_date", "num_guests"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_reservations",
            "description": "Kayıtlı rezervasyonları getirir. Misafir kendi rezervasyonunu sormak istediğinde çağır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "guest_name": {
                        "type": "string",
                        "description": "Misafirin adı soyadı. Belirtilmezse tüm rezervasyonlar döner."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extend_reservation",
            "description": "Mevcut bir rezervasyonun çıkış tarihini uzatır. Misafir rezervasyonunu uzatmak istediğinde önce get_reservations ile rezervasyonu bul, sonra bu aracı çağır.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reservation_id": {
                        "type": "string",
                        "description": "Rezervasyon ID'si (get_reservations'dan alınır)"
                    },
                    "new_checkout_date": {
                        "type": "string",
                        "description": "Yeni çıkış tarihi YYYY-MM-DD formatında"
                    }
                },
                "required": ["reservation_id", "new_checkout_date"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# Tool çağrısını çalıştıran fonksiyon
# ---------------------------------------------------------------------------

def get_reservations(guest_name: str = None) -> dict:
    """Kayıtlı rezervasyonları döndürür. guest_name verilirse sadece o misafirin rezervasyonları."""
    reservations = load_reservations()
    if guest_name:
        filtered = [r for r in reservations if guest_name.lower() in r.get("guest_name", "").lower()]
    else:
        filtered = reservations
    return {"reservations": filtered, "total": len(filtered)}


def extend_reservation(reservation_id: str, new_checkout_date: str) -> dict:
    """Mevcut bir rezervasyonun çıkış tarihini uzatır."""
    reservations = load_reservations()

    target = None
    for r in reservations:
        if r.get("id") == reservation_id:
            target = r
            break

    if not target:
        return {"success": False, "message": f"Rezervasyon bulunamadı: {reservation_id}"}

    try:
        new_checkout = datetime.strptime(new_checkout_date, "%Y-%m-%d").date()
        old_checkout = datetime.strptime(target["checkout_date"], "%Y-%m-%d").date()
        checkin = datetime.strptime(target["checkin_date"], "%Y-%m-%d").date()
    except ValueError:
        return {"success": False, "message": "Tarih formatı hatalı. YYYY-MM-DD olmalı."}

    if new_checkout <= old_checkout:
        return {"success": False, "message": "Yeni çıkış tarihi mevcut çıkış tarihinden sonra olmalıdır."}

    # Uzatılan günler için müsaitlik kontrolü
    availability = check_availability(old_checkout.strftime("%Y-%m-%d"), new_checkout_date, target["room_type"])
    if "error" in availability:
        return {"success": False, "message": availability["error"]}

    room_avail = availability["availability"].get(target["room_type"], {})
    if room_avail.get("available", 0) <= 0:
        return {"success": False, "message": "Üzgünüz, bu tarihler için oda müsait değil."}

    # Rezervasyonu güncelle
    new_nights = (new_checkout - checkin).days
    new_total = new_nights * target["price_per_night"]

    target["checkout_date"] = new_checkout_date
    target["nights"] = new_nights
    target["total_price"] = new_total

    save_reservations(reservations)
    logger.info("Rezervasyon uzatıldı: %s → %s", reservation_id, new_checkout_date)

    return {
        "success": True,
        "message": "Rezervasyon başarıyla uzatıldı.",
        "reservation_id": reservation_id,
        "guest_name": target["guest_name"],
        "room_type": target["room_type"],
        "checkin_date": target["checkin_date"],
        "new_checkout_date": new_checkout_date,
        "total_nights": new_nights,
        "total_price": new_total
    }


def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Asistanın çağırdığı tool'u çalıştırır ve sonucu string olarak döner."""
    logger.debug("Tool çağrıldı: %s, args: %s", tool_name, tool_args)

    if tool_name == "check_availability":
        result = check_availability(**tool_args)
    elif tool_name == "make_reservation":
        result = make_reservation(**tool_args)
    elif tool_name == "get_reservations":
        result = get_reservations(**tool_args)
    elif tool_name == "extend_reservation":
        result = extend_reservation(**tool_args)
    else:
        result = {"error": f"Bilinmeyen tool: {tool_name}"}

    logger.debug("Tool sonucu: %s", result)
    return json.dumps(result, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Agent döngüsü — tool call'ları yönetir
# ---------------------------------------------------------------------------

def run_agent(client: OpenAI, messages: list[dict]) -> str:
    """
    Agent döngüsü: asistan tool çağırırsa çalıştırır, sonucu gönderir, tekrar sorar.
    Asistan düz metin döndürene kadar devam eder.
    """
    full_response = ""

    max_iterations = 10  # Sonsuz döngüyü önlemek için maksimum iterasyon
    iteration = 0
    try:
        while iteration < max_iterations:
            iteration += 1
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                stream=False  # Tool call varsa streaming kullanamıyoruz
            )

            message = response.choices[0].message

            # Asistan tool çağırdı mı?
            if message.tool_calls:
                # Tool çağrılarını messages listesine ekle
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                # Her tool'u çalıştır ve sonucu ekle
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_result = execute_tool(tool_name, tool_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })

                # Döngü devam ediyor — asistan tool sonuçlarına bakarak cevap verecek

            else:
                # Asistan düz metin döndürdü
                full_response = message.content or ""

                # Model bazen tool çağrısını eski formatta düz metin olarak yazabiliyor
                # Format 1: <function=name({"key": "val"})></function>
                # Format 2: <function=name={"key": "val"}</function>
                malformed_pattern = re.search(
                    r'<function=(\w+)[=(](\{.*?\})\)?</function>',
                    full_response, re.DOTALL
                )
                if malformed_pattern:
                    tool_name = malformed_pattern.group(1)
                    try:
                        tool_args = json.loads(malformed_pattern.group(2))
                    except Exception:
                        tool_args = {}

                    logger.debug("Hatalı format tool yakalandı: %s %s", tool_name, tool_args)
                    tool_result = execute_tool(tool_name, tool_args)

                    # Tool sonucunu mesaj geçmişine ekleyip döngüyü devam ettir
                    messages.append({"role": "assistant", "content": full_response})
                    messages.append({"role": "user", "content": f"[Tool sonucu - {tool_name}]: {tool_result}. Lütfen bu bilgiye dayanarak misafire cevap ver."})
                    continue

                # Normal temizlik
                clean_response = re.sub(r'<function=\w+[^>]*>.*?</function>', '', full_response, flags=re.DOTALL).strip()
                clean_response = re.sub(r'\[Misafirin dili:.*?\]', '', clean_response).strip()

                if clean_response:
                    print("\n🏨 Resepsiyon: ", end="", flush=True)
                    words = clean_response.split(" ")
                    for word in words:
                        print(word + " ", end="", flush=True)
                    print("\n")
                    full_response = clean_response

                # Son cevabı messages listesine ekle
                messages.append({"role": "assistant", "content": full_response})
                break

        if iteration >= max_iterations:
            logger.warning("Agent döngüsü maksimum iterasyona ulaştı.")
            print("\n⚠️  Üzgünüm, isteğinizi işleyemedim. Lütfen tekrar deneyin.\n")

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

# ---------------------------------------------------------------------------
# Konuşma geçmişini kaydet
# ---------------------------------------------------------------------------

def save_conversation(messages: list[dict], session_id: str):
    """Oturum bitince konuşmayı conversation_history.json dosyasına kaydeder."""
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    # System prompt ve tool mesajlarını hariç tut, sadece kullanıcı/asistan konuşmasını kaydet
    clean_messages = [
        m for m in messages
        if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str) and m.get("content")
    ]

    conversation = {
        "session_id": session_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": clean_messages
    }

    history.append(conversation)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    logger.info("Konuşma geçmişi kaydedildi: %s", HISTORY_FILE)

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    if not API_KEY:
        logger.error("OPENAI_API_KEY bulunamadı.")
        print("\n❌  API anahtarı bulunamadı. Lütfen .env dosyanızı kontrol edin.\n")
        sys.exit(1)
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


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


def show_reservations():
    """Kayıtlı tüm rezervasyonları terminalde gösterir."""
    reservations = load_reservations()
    if not reservations:
        print("\n📋 Henüz kayıtlı rezervasyon yok.\n")
        return
    print(f"\n📋 Toplam {len(reservations)} rezervasyon:\n")
    for r in reservations:
        print(f"  [{r.get('created_at', '?')}]")
        print(f"  Misafir  : {r.get('guest_name', '?')}")
        print(f"  Oda      : {r.get('room_type', '?')}")
        print(f"  Giriş    : {r.get('checkin_date', '?')}")
        print(f"  Çıkış    : {r.get('checkout_date', '?')}")
        print(f"  Toplam   : {r.get('total_price', '?')} TL")
        print()

# ---------------------------------------------------------------------------
# Ana döngü
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """Metnin dilini Unicode ve kelime analizi ile tespit eder. API çağrısı yapmaz."""
    text_lower = text.lower().strip()

    # Arapça karakterler
    if any("؀" <= c <= "ۿ" for c in text):
        return "Arapça"

    # Türkçe özel karakterler varsa büyük ihtimalle Türkçe
    turkce_chars = set("çğışöüÇĞİŞÖÜ")
    if any(c in turkce_chars for c in text):
        return "Türkçe"

    # Almanca özel karakterler
    almanca_chars = set("äöüßÄÖÜ")
    if any(c in almanca_chars for c in text):
        return "Almanca"

    # Fransızca özel karakterler
    fransizca_chars = set("àâæéèêëîïôœùûüÿÀÂÆÉÈÊËÎÏÔŒÙÛÜŸ")  # ç kaldırıldı, Türkçe ile çakışıyordu
    if any(c in fransizca_chars for c in text):
        return "Fransızca"

    # Yaygın İngilizce kelimeler
    english_words = {"hello", "hi", "hey", "good", "morning", "evening", "please",
                     "thank", "thanks", "yes", "no", "what", "how", "i", "we",
                     "need", "want", "room", "book", "reservation", "can", "could",
                     "would", "speak", "do", "have", "is", "are", "my", "the", "a"}

    # Yaygın Almanca kelimeler
    german_words = {"guten", "hallo", "bitte", "danke", "ja", "nein", "zimmer",
                    "ich", "bin", "sie", "haben", "ist", "und", "oder", "mit",
                    "ein", "der", "die", "das", "wie", "was", "wir", "nicht"}

    # Yaygın Fransızca kelimeler
    french_words = {"bonjour", "bonsoir", "merci", "oui", "non", "je", "vous",
                    "nous", "est", "une", "les", "des", "avec", "pour", "chambre"}

    # Yaygın Türkçe kelimeler
    turkish_words = {"merhaba", "selam", "evet", "hayır", "teşekkür", "lütfen",
                     "nasıl", "iyi", "tamam", "rezervasyon", "oda", "gün", "gece",
                     "yarın", "bugün", "istiyorum", "var", "yok", "bilgi", "almak"}

    words = set(text_lower.split())

    scores = {
        "Türkçe": len(words & turkish_words),
        "İngilizce": len(words & english_words),
        "Almanca": len(words & german_words),
        "Fransızca": len(words & french_words),
    }

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        logger.debug("Tespit edilen dil: %s (skor: %s)", best, scores)
        return best

    return "Türkçe"


def main():
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print_banner(session_id)

    client = get_client()

    # Program açılınca geçmiş rezervasyonları arşivle
    archive_past_reservations()

    messages: list[dict] = [{"role": "system", "content": build_system_prompt()}]

    logger.info("Oturum başladı. ID: %s | Model: %s", session_id, MODEL)

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

        if user_input.lower() == "rezervasyonlar":
            show_reservations()
            continue

        # Misafirin dilini Unicode analizi ile tespit edip mesaja ekle — asistan doğru dilde cevap versin
        detected_lang = detect_language(user_input)
        enriched_input = f"[Misafirin dili: {detected_lang}] {user_input}"
        messages.append({"role": "user", "content": enriched_input})
        logger.debug("Misafir: %s", user_input)

        # Agent döngüsünü çalıştır — tool call varsa otomatik halleder
        assistant_reply = run_agent(client, messages)

        if assistant_reply:
            logger.debug("Resepsiyon: %s", assistant_reply.strip())


if __name__ == "__main__":
    main()
