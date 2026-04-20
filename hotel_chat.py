#!/usr/bin/env python3
# Eski sürümde regex ile string parçalamaya çalışıyordum
# Şimdi doğrudan OpenAI SDK kullanıyor

import os
import sys
import json
import logging
# import time  # Lazım olur diye ekledim ama kullanmadım şimdilik kalsın
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, set_default_openai_client, set_default_openai_api, ModelSettings
from agents.exceptions import MaxTurnsExceeded

# API key'leri Github'a public pushlayıp patlamamak için dotenv kullanıyoruz
load_dotenv()

# --- Değişkenler ---
# Dosya yolları
LOG_FILE = "hotel_chat.log"
RESERVATIONS_FILE = "reservations.json"
ARCHIVE_FILE = "reservations_archive.json"

MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("MODEL_API_BASE_URL", "https://api.groq.com/openai/v1")

# Odaları ve fiyatları dict içine attım.
# İleride belki veritabanına bağlarım (üşenmezsem)
ROOM_CAPACITY = {"standart": 10, "deluxe": 12, "suite": 8, "apart": 7}
ROOM_PRICES = {"standart": 4500, "deluxe": 5500, "suite": 7500, "apart": 9000}


# --- SDK Ayarları ---
# Groq client'ı SDK'ya yedirme taktiği. Dokümantasyon okumaktan gözüm çıktı bunu bulana kadar.
groq_client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
set_default_openai_client(groq_client)
set_default_openai_api("chat_completions") 

# SDK'nın gereksiz loglarını kapatıyorum, terminali çok pisletiyor
from agents import set_tracing_disabled
set_tracing_disabled(True)

def setup_logging():
    # Model arka planda saçmalarsa logdan bakıp bulmak için
    logger = logging.getLogger("hotel-cli")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING) # Terminale sadece warning ve error bassın
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()


# --- JSON İşlemleri ---

def load_reservations():
    # Dosya yoksa crash yemesin diye try-except koydum.
    # İlk başta burada çok patlıyordu program.
    if not os.path.exists(RESERVATIONS_FILE):
        return []
    try:
        with open(RESERVATIONS_FILE, "r", encoding="utf-8") as f:
            all_reservations = json.load(f)
        
        # İçinde eksik veri olan bozuk kayıtlar varsa onları eliyoruz (filter gibi)
        req_fields = {"guest_name", "room_type", "checkin_date", "checkout_date"}
        temp_list = []
        for r in all_reservations:
            if req_fields.issubset(r.keys()):
                temp_list.append(r)
        return temp_list
    except Exception as e:
        # print(f"Okurken hata oldu: {e}") # Debug için koymuştum
        return []

def save_reservations(reservations):
    # Türkçe karakterler json'da unicode ascii olarak görünmesin diye ensure_ascii=False
    with open(RESERVATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(reservations, f, ensure_ascii=False, indent=2)

def archive_past_reservations():
    # Program her açıldığında tarihi geçmiş rezervasyonları arşive şutluyoruz.
    # Yoksa json dosyası ileride çok şişer. (Big Data xd)
    reservations = load_reservations()
    today = date.today()
    active_res = []
    archived_res = []

    for r in reservations:
        try:
            checkout = datetime.strptime(r["checkout_date"], "%Y-%m-%d").date()
            if checkout < today:
                archived_res.append(r)
            else:
                active_res.append(r)
        except (KeyError, ValueError):
            # Tarihi parse edemezsek aktifte bırakıyorum şimdilik
            active_res.append(r) 

    if len(archived_res) == 0:
        return # Arşivlenecek bir şey yoksa çık

    existing_archive = []
    if os.path.exists(ARCHIVE_FILE):
        try:
            with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                existing_archive = json.load(f)
        except Exception:
            pass # okuyamazsa boş liste kalsın napalım

    existing_archive.extend(archived_res)
    with open(ARCHIVE_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_archive, f, ensure_ascii=False, indent=2)

    save_reservations(active_res)
    logger.info(f"{len(archived_res)} tane eski kayıt arşivlendi.")
    print(f"\n📦 {len(archived_res)} eski rezervasyon arşivlendi.\n")


# --- TOOL FONKSİYONLARI ---
# Buraları @function_tool ile sarmalıyoruz ki model bunları kullanabilsin.

@function_tool
def check_availability(checkin_date: str, checkout_date: str, room_type: str = "") -> str:
    """
    Belirli tarihler arası oda sayıyor. YYYY-MM-DD istiyor her zaman.
    """
    if room_type == "":
        room_type = None

    try:
        checkin = datetime.strptime(checkin_date, "%Y-%m-%d").date()
        checkout = datetime.strptime(checkout_date, "%Y-%m-%d").date()
    except ValueError:
        return json.dumps({"error": "Tarih formatı patladı hacı, YYYY-MM-DD gönder."})

    if checkin >= checkout:
        return json.dumps({"error": "Çıkış girişten önce olamaz mantıken."})

    if checkin < date.today():
        return json.dumps({"error": "Zaman makinemiz yok, geçmişe rezervasyon yapamayız :)"})

    reservations = load_reservations()
    booked_counts = {}
    
    # İstenen tarihler arasındaki her gün için bir sayaç açıyoruz
    current = checkin
    while current < checkout:
        date_str = current.strftime("%Y-%m-%d")
        booked_counts[date_str] = {"standart": 0, "deluxe": 0, "suite": 0, "apart": 0}
        current += timedelta(days=1)

    # Burası biraz spagetti, Big O(n^2) falan oldu galiba ama 
    # alt tarafı 37 oda var, optimizasyon kasmaya gerek yok bence çalışıyor sonuçta :D
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
        except Exception:
            continue

    availability = {}
    for room, capacity in ROOM_CAPACITY.items():
        if room_type and room_type.lower() != room:
            continue
        min_available = capacity
        for date_str in booked_counts:
            # O günkü boş oda sayısını bul
            available = capacity - booked_counts[date_str].get(room, 0)
            if available < min_available:
                min_available = available
                
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
def make_reservation(guest_name: str, room_type: str, checkin_date: str, checkout_date: str, num_guests: int) -> str:
    """
    JSON'a rezervasyon basan fonksiyon.
    """
    avail_result = _check_availability_internal(checkin_date, checkout_date, room_type)

    if "error" in avail_result:
        return json.dumps({"success": False, "message": avail_result["error"]})

    room_key = room_type.lower()
    if room_key not in avail_result["availability"]:
        return json.dumps({"success": False, "message": "Otelde öyle bir oda yok malesef."})

    if avail_result["availability"][room_key]["available"] <= 0:
        return json.dumps({"success": False, "message": "O tarihlerde odalar ful çekiyor."})

    nights = avail_result["nights"]
    price_per_night = ROOM_PRICES.get(room_key, 0)

    # UUID import etmeye üşendim, anlık tarihi string yapıp ID diye yediriyorum xd
    rez_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    reservation = {
        "id": rez_id,
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
    logger.info(f"Yazıldı: {guest_name} - {room_type} - {checkin_date}")

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
        "message": "Rezervasyon tamam."
    }, ensure_ascii=False)


@function_tool
def get_reservations(guest_name: str = "") -> str:
    """İsimden rezervasyon buluyor."""
    reservations = load_reservations()
    sonuclar = []
    if guest_name:
        for r in reservations:
            # küçük harfe çevirip aratıyoruz case sensitive patlamasın diye
            if guest_name.lower() in r.get("guest_name", "").lower():
                sonuclar.append(r)
        reservations = sonuclar
    return json.dumps({"reservations": reservations, "total": len(reservations)}, ensure_ascii=False)


@function_tool
def extend_reservation(reservation_id: str, new_checkout_date: str) -> str:
    """Günü uzatmak isteyenler için."""
    reservations = load_reservations()
    target = None
    for r in reservations:
        if r.get("id") == reservation_id:
            target = r
            break

    if not target:
        return json.dumps({"success": False, "message": "ID hatalı, bulamadım."})

    try:
        new_checkout = datetime.strptime(new_checkout_date, "%Y-%m-%d").date()
        old_checkout = datetime.strptime(target["checkout_date"], "%Y-%m-%d").date()
        checkin = datetime.strptime(target["checkin_date"], "%Y-%m-%d").date()
    except ValueError:
        return json.dumps({"success": False, "message": "Tarih formatında bi gariplik var."})

    if new_checkout <= old_checkout:
        return json.dumps({"success": False, "message": "Zaten o gün çıkıyorsun, daha ileri bi tarih seç."})

    # Aradaki uzatılan günlerin müsaitliğini kontrol ediyoruz (sessiz fonksiyonla)
    avail_result = _check_availability_internal(old_checkout.strftime("%Y-%m-%d"), new_checkout_date, target["room_type"])
    if "error" in avail_result:
        return json.dumps({"success": False, "message": avail_result["error"]})

    if avail_result["availability"].get(target["room_type"], {}).get("available", 0) <= 0:
        return json.dumps({"success": False, "message": "Uzatmak istediğin günlerde odalar dolu maalesef."})

    new_nights = (new_checkout - checkin).days
    target["checkout_date"] = new_checkout_date
    target["nights"] = new_nights
    target["total_price"] = new_nights * target["price_per_night"]
    
    save_reservations(reservations)
    return json.dumps({
        "success": True,
        "message": "Uzatıldı.",
        "reservation_id": reservation_id,
        "new_checkout_date": new_checkout_date,
        "total_nights": new_nights,
        "total_price": target["total_price"]
    }, ensure_ascii=False)


# --- Yardımcı / İç Fonksiyonlar ---

def _check_availability_internal(checkin_date, checkout_date, room_type=None):
    # Model tool içinden tool çağırmasın diye aynı mantığın arka plan versiyonunu yaptım.
    # Kod tekrarı oldu biraz ama idare eder.
    if room_type == "": room_type = None

    try:
        checkin = datetime.strptime(checkin_date, "%Y-%m-%d").date()
        checkout = datetime.strptime(checkout_date, "%Y-%m-%d").date()
    except ValueError:
        return {"error": "Format hatası"}

    if checkin >= checkout: return {"error": "Tarih mantıksız"}
    if checkin < date.today(): return {"error": "Geçmiş zaman"}

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
        except Exception:
            pass

    availability = {}
    for room, capacity in ROOM_CAPACITY.items():
        if room_type and room_type.lower() != room: continue
        min_avail = capacity
        for date_str in booked_counts:
            min_avail = min(min_avail, capacity - booked_counts[date_str].get(room, 0))
        availability[room] = {
            "capacity": capacity, "available": min_avail, "price_per_night": ROOM_PRICES[room]
        }
    return {"checkin_date": checkin_date, "checkout_date": checkout_date, "nights": (checkout - checkin).days, "availability": availability}


def detect_language(text: str) -> str:
    # Model bazen dili karıştırıyor (loglarda Sagen falan yazmıştı hatırlarsan). 
    # Normalde buraya spacy falan kurulur da 2 kelime için kütüphane kasmak istemedim.
    # Bodoslama if-else check yapıyorum.
    if any(c in "çğışöüÇĞİŞÖÜ" for c in text): return "Türkçe"
    if any(c in "äöüßÄÖÜ" for c in text): return "Almanca"
    
    english_words = ["hello", "hi", "hey", "good", "please", "yes", "no", "room"]
    german_words = ["hallo", "bitte", "danke", "ja", "nein", "ich", "bin"]
    turkish_words = ["merhaba", "selam", "evet", "hayır", "lütfen", "nasıl", "oda"]

    metin = text.lower()
    
    tr_count = sum(1 for w in turkish_words if w in metin)
    en_count = sum(1 for w in english_words if w in metin)
    de_count = sum(1 for w in german_words if w in metin)
    
    if en_count > tr_count and en_count > de_count: return "İngilizce"
    if de_count > tr_count and de_count > en_count: return "Almanca"
    
    return "Türkçe" # default


def build_instructions():
    # Promptu dinamik basıyorum ki model bugünü bilsin.
    now = datetime.now()
    yarın = now + timedelta(days=1)
    
    return f"""Sen Renata Suites Boutique Hotel'in asistanısın.

BUGÜN: {now.strftime("%d %B %Y")} — Saat: {now.strftime("%H:%M")}
YARIN: {yarın.strftime("%d %B %Y")}

KURAL: Misafir nece yazıyorsa o dilde cevap ver. Karıştırma.

ÇOK ÖNEMLİ:
- make_reservation çağırmadan önce kesinlikle şu bilgileri topla:
  1. Ad Soyad
  2. Oda tipi
  3. Giriş tarihi
  4. Çıkış tarihi  
  5. Misafir sayısı
- Eksik bilgi varsa toolu kullanma, misafire sor.
- guest_name alanına kafandan isim uydurma.

Otel Adres: Nakiye Elgün Sk. No:44, Osmanbey/Şişli
Odalar: Standart (4500 TL), Deluxe (5500 TL), Suite (7500 TL), Apart (9000 TL)
"""

def show_reservations():
    # Hızlıca terminalden db kontrol etmek için
    reservations = load_reservations()
    if not reservations:
        print("\n📋 Kayıt yok.\n")
        return
    print(f"\n📋 Aktif Rezervasyonlar ({len(reservations)} tane):\n")
    for r in reservations:
        print(f"  {r.get('guest_name', '?')} | {r.get('room_type', '?')} | {r.get('checkin_date', '?')} | {r.get('total_price', '?')} TL")
    print()


# --- ANA DÖNGÜ ---

def main():
    if not API_KEY:
        print("\n❌ Kanka .env dosyasında key yok, patlar bu.\n")
        sys.exit(1)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Başlarken bi çöpleri dökelim
    archive_past_reservations()

    print("=" * 55)
    print("  🏨 Renata Suites Boutique Hotel")
    print(f"  Model : {MODEL}")
    print("  Çıkmak için 'exit' yaz. Liste için 'rezervasyonlar'.")
    print("=" * 55)
    print("\nHoş geldiniz!\n")

    agent = Agent(
        name="Renata Asistan",
        model=MODEL,
        instructions=build_instructions(),
        tools=[check_availability, make_reservation, get_reservations, extend_reservation],
        model_settings=ModelSettings(parallel_tool_calls=False)
    )

    conversation_history = []

    while True:
        try:
            user_input = input("💬 Misafir: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Kapatılıyor...\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "çık"]:
            print("\n👋 Kapatılıyor...\n")
            break

        if user_input.lower() == "rezervasyonlar":
            show_reservations()
            continue

        # Dil taktiğini buraya yapıştırdım
        lang = detect_language(user_input)
        gizli_prompt = f"[Misafirin dili: {lang}] {user_input}"
        
        conversation_history.append({"role": "user", "content": gizli_prompt})

        # Model timeout atarsa falan diye 2 deneme hakkı var
        for deneme in range(2):
            try:
                result = Runner.run_sync(agent, conversation_history, max_turns=10)
                cevap = result.final_output
                print(f"\n🏨 Resepsiyon: {cevap}\n")
                
                # Burda sadece son metni değil, modelin toolcall adımlarını da hafızaya
                # almak için geçmişi doğrudan SDK'nın döndürdüğü tam liste ile güncelliyorum:
                if hasattr(result, 'messages'):
                    conversation_history = result.messages
                elif hasattr(result, 'history'):
                    conversation_history = result.history
                else:
                    # Eğer SDK conversation historyi kendiliğinden güncelliyorsa,
                    # append satırını silmek bile yeterli olur.
                    pass 
                
                break 

            except MaxTurnsExceeded:
                print("\n⚠️ Anlayamadım, baştan yazar mısın?\n")
                break
            except Exception as e:
                hata_msaji = str(e)
                # Model tool kullanmayı beceremezse context'i yenileyip şans veriyoruz
                if "tool_use_failed" in hata_msaji or "tool call validation" in hata_msaji:
                    if deneme == 0 and conversation_history and conversation_history[-1]["role"] == "user":
                        # Son mesajı çek çıkar yap ki model kendine gelsin
                        conversation_history.append(conversation_history.pop())
                        continue
                    else:
                        print("\n⚠️ API'de bi anlık sıkıntı oldu galiba, tekrar yazar mısın.\n")
                else:
                    print(f"\n❌ Beklenmeyen hata: {e}\n")
                break

if __name__ == "__main__":
    main()