# 🤖 LLM CLI Chat

A lightweight Python command-line chat application that connects to OpenAI-compatible LLM APIs. Supports streaming responses, full conversation history, `.env`-based API key management, and structured logging.

---

## ✨ Features

| Feature | Status |
|---|---|
| Static system prompt | ✅ |
| Multi-turn conversation (history) | ✅ |
| Streaming responses | ✅ |
| `.env` API key management | ✅ |
| Logging to file (`chat.log`) | ✅ |
| Graceful error handling | ✅ |
| `exit` / `quit` / Ctrl+C to end session | ✅ |

---

## 📦 Kurulum

### 1. Repoyu klonlayın

```bash
git clone https://github.com/kullanici-adi/llm-cli.git
cd llm-cli
```

### 2. (Önerilen) Sanal ortam oluşturun

```bash
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
```

### 3. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

### 4. API anahtarınızı tanımlayın

```bash
cp .env.example .env
# .env dosyasını açıp OPENAI_API_KEY değerini girin
```

`.env` dosyası örneği:

```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini   # opsiyonel, varsayılan: gpt-4o-mini
```

---

## 🚀 Çalıştırma

```bash
python chat.py
```

### Örnek oturum

```
=======================================================
  🤖  LLM CLI Chat  —  powered by OpenAI
  Model : gpt-4o-mini
  Log   : chat.log
  Çıkmak için 'exit' veya 'quit' yazın.
=======================================================

💬 Siz: Python'da liste ile tuple arasındaki fark nedir?

🤖 Asistan: Liste (list) değiştirilebilir (mutable) bir veri yapısıdır...

💬 Siz: exit

👋  Görüşmek üzere!
```

---

## 📚 Kullanılan Kütüphaneler

| Kütüphane | Versiyon | Amaç |
|---|---|---|
| `openai` | ≥ 1.30.0 | OpenAI API istemcisi (streaming dahil) |
| `python-dotenv` | ≥ 1.0.0 | `.env` dosyasından ortam değişkeni yükleme |

> Yalnızca Python standart kütüphanesi (`os`, `sys`, `logging`) ve bu iki paket kullanılmıştır.

---

## 🗂️ Proje Yapısı

```
llm-cli/
├── chat.py            # Ana uygulama
├── requirements.txt   # Bağımlılıklar
├── .env.example       # Örnek ortam değişkenleri
├── .env               # Gerçek API anahtarı (git'e eklenmez)
├── .gitignore
└── README.md
```

---

## 🔒 Güvenlik Notları

- `.env` dosyası **asla** versiyon kontrolüne eklenmemelidir (`.gitignore`'a eklenmiştir).
- `chat.log` dosyası konuşma geçmişini içerir; paylaşırken dikkatli olun.

---

## 🛠️ Geliştirme Notları

- **Model değiştirme:** `.env` dosyasında `LLM_MODEL` değişkenini ayarlayın.
- **System prompt:** `chat.py` içindeki `SYSTEM_PROMPT` sabitini düzenleyin.
- **Log seviyesi:** `setup_logging()` fonksiyonundaki `fh.setLevel` ile kontrol edilebilir.
