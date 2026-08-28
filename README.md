# Otel Resepsiyon Asistanı

Bir otelin resepsiyon görevlisi gibi konuşan, gerçekten rezervasyon yapabilen komut satırı asistanı. Misafir serbest metinle yazıyor — "yarından sonraki güne iki kişilik bir suite" — asistan tarihi çözüyor, müsaitliği kontrol ediyor, fiyatı hesaplıyor ve rezervasyonu kaydediyor.

Konuşmayı bir LLM yürütüyor ama işi **tool-calling** yapıyor: model müsaitliği tahmin etmiyor, `check_availability` fonksiyonunu çağırıp gerçek doluluk tablosuna bakıyor. Fiyat, kapasite ve çakışma kontrolü Python tarafında; modelin uydurma alanı yok.

```
Misafir:    yarından sonraki güne rezervasyon yapmak istiyorum,
            oda tiplerinizi öğrenebilir miyim?
→ tool:     check_availability(checkin="2026-03-21", checkout="2026-03-22")
← sonuç:    standart 10/10 · deluxe 12/12 · suite 8/8 · apart 7/7
Resepsiyon: Standart 4500 TL, Deluxe 5500 TL, Suite 7500 TL, Apart 9000 TL.
            Tüm oda tiplerimizde müsaitlik var.
```

## Ne yapabiliyor

| Araç | İş |
|---|---|
| `check_availability` | Tarih aralığında oda tipi bazında boşluk ve fiyat |
| `make_reservation` | Kapasite kontrolüyle rezervasyon oluşturma |
| `get_reservations` | Misafir adına kayıt sorgulama |
| `extend_reservation` | Konaklama uzatma, çakışma kontrolüyle |

Bunların üstünde iki tane daha var: gelen mesajın dilini tespit edip aynı dilde cevap veren bir katman, ve çıkış tarihi geçmiş kayıtları otomatik arşive taşıyan bir temizlik adımı.

## Mimari

- **`agents`** (OpenAI Agents SDK) — araçlar `@function_tool` ile tanımlı, ajan döngüsünü Runner yürütüyor, `MaxTurnsExceeded` ile sonsuz döngü koruması var.
- **Model** — Groq üzerinden `llama-3.3-70b-versatile`. OpenAI uyumlu endpoint kullanıldığı için `MODEL_API_BASE_URL` değiştirilerek başka bir sağlayıcıya geçilebilir.
- **Durum** — JSON dosyalarında; rezervasyonlar ve arşiv ayrı. Veritabanı yok, kasıtlı: proje LLM'e iş yaptırma kısmına odaklı.
- **Log** — her oturum ayrı ID ile; misafir mesajı, çağrılan araç, aracın döndürdüğü sonuç ve modelin cevabı ayrı ayrı kaydediliyor. Modelin neden öyle cevap verdiğini geriye dönük okuyabilmek için.

## Çalıştırma

```bash
cp .env.example .env          # ve Groq anahtarını içine yaz
python -m pip install -r requirements.txt
python hotel_chat.py
```

Çıkmak için `exit`.

## Notlar

Rezervasyon ve konuşma dosyaları çalışma zamanında üretiliyor, repoda tutulmuyor. `.env` hiçbir zaman commit'lenmez; API anahtarı yalnızca ortam değişkeninden okunur.

---

© 2026 Yavuz Selim Şeremetli
