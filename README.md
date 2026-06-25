<div align="center">

<h1>⚖️ HAQ</h1>

<p><strong>Pakistan's first RAG-based AI legal assistant — multi-turn chat, document analysis, legal notice generation & location-based legal help for citizens who cannot afford lawyers.</strong></p>

<p>
  <a href="https://github.com/Shahrukh-aidev/Advocate-AI/stargazers"><img src="https://img.shields.io/github/stars/Shahrukh-aidev/Advocate-AI?style=for-the-badge&color=FFD700" alt="Stars"/></a>
  <a href="https://github.com/Shahrukh-aidev/Advocate-AI/network/members"><img src="https://img.shields.io/github/forks/Shahrukh-aidev/Advocate-AI?style=for-the-badge&color=4FC3F7" alt="Forks"/></a>
  <a href="https://github.com/Shahrukh-aidev/Advocate-AI/issues"><img src="https://img.shields.io/github/issues/Shahrukh-aidev/Advocate-AI?style=for-the-badge&color=FF7043" alt="Issues"/></a>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/AI-Groq%20Powered-blueviolet?style=for-the-badge" alt="Groq"/>
  <img src="https://img.shields.io/badge/RAG-Pinecone%20%2B%20Cohere-orange?style=for-the-badge" alt="RAG"/>
  <img src="https://img.shields.io/badge/Deployed-Hugging%20Face%20Spaces-blue?style=for-the-badge" alt="Spaces"/>
</p>

<p>
  <a href="#-features">Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-screenshots">Screenshots</a> •
  <a href="#-contributing">Contributing</a>
</p>

</div>

---

## 🧠 What is HAQ?

**HAQ** (meaning *"Right"* in Urdu) is Pakistan's first **RAG-based AI legal assistant** designed to democratize access to justice. It empowers everyday citizens — especially those who cannot afford lawyers — with accurate, actionable legal information in **Urdu and English**.

From asking a simple legal question to receiving a structured answer with exact law citations, uploading a court document for AI analysis, or generating a professionally formatted legal notice with official verification links — the entire pipeline is automated, intelligent, and completely free.

> Built by [Shahrukh Hussain](https://github.com/Shahrukh-aidev) — AI/ML Developer & CS Student at Sukkur IBA University 🇵🇰

---

## ✨ Features

### 💬 Multi-Turn Legal Chat
- **Contextual Memory** — HAQ remembers your full conversation across 10+ turns. Ask a question, get an answer, then follow up with *"What if he doesn't comply?"* or *"Phir kya hoga?"* — no need to repeat yourself.
- **Bilingual Support** — Ask in Urdu or English; HAQ matches your language with pure Urdu (no Hindi) for legal accuracy.
- **Structured Answers** — Every response follows: Legal Basis → The Ruling → What You Should Do → Where to Go → Disclaimer.
- **Safety Guardrails** — Built-in filters reject harmful, illegal, or jailbreak prompts.

### 🎙️ Voice Input & Voice Output
- **Voice Input** — Speak your legal question in Urdu or English; HAQ transcribes it via Groq Whisper (large-v3) with high accuracy.
- **Voice Output** — Listen to HAQ's answers in your language using Google TTS — perfect for low-literacy users or when you're on the go.
- **Auto Language Detection** — Automatically detects Urdu vs English for TTS playback.

### 📄 Document Upload & Analysis
- **Upload any legal document** — court notices, FIR copies, contracts, property deeds, rent agreements, bank documents, or legal notices.
- **Smart Extraction** — Uses `pdfplumber` for native PDF text extraction and **Tesseract OCR** for scanned images and image-based PDFs (English + Urdu support).
- **AI Analysis** — HAQ explains what the document means, highlights critical clauses, identifies deadlines, explains legal implications, suggests next steps, and flags red flags.

### ⚖️ Legal Notice Generator
- Generate a **professional Pakistani legal notice** with exact law citations, reference number, and 15-day standard deadline.
- **Party Structure** — FROM (Murasil / Noticee) → TO (Mukhatib / Respondent).
- **Download as PDF** (print-ready) or **Word (.docx)** (editable) — saves ₨5,000–10,000 in lawyer fees.
- Includes **Acknowledgement of Receipt** section for legal proof.

### 📍 Location-Based Legal Help
- **Auto-detect** your city via browser geolocation or select manually from 21+ major Pakistani cities.
- **Local Resources** — nearest High Court, District Courts, Police Stations, Free Legal Aid Centres (with phone numbers), and local Bar Association contacts.
- **Provincial Law Variations** — explains how laws differ across Sindh, Punjab, KPK, Balochistan, ICT, AJK, and Gilgit-Baltistan.

### 🔗 Official Law Verification
- Every answer includes **clickable verification cards** linking directly to **pakistancode.gov.pk** (Pakistan Ministry of Law & Justice).
- **14 Major Laws** auto-detected: Constitution 1973, PPC 1860, CrPC 1898, PECA 2016, MFLO 1961, Contract Act 1872, TPA 1882, and more.

### 🎨 Premium Dark UI
- Custom-designed dark theme with gold accents inspired by Pakistani legal heritage.
- Fully responsive for mobile and desktop.

---

## 🛠 Tech Stack

### Backend
| Layer | Technology |
|-------|-----------|
| Framework | Python + Gradio 6.10 |
| LLM | Groq API (Llama 3.1-8B / 3.3-70B, Gemma 2-9B) |
| Embeddings | Cohere (embed-english-light-v3.0) |
| Vector DB | Pinecone (haq-law index, 4000+ laws) |
| Speech-to-Text | Groq Whisper (large-v3) |
| Text-to-Speech | gTTS (Google TTS) |
| OCR | Tesseract + pdf2image + pdfplumber |
| PDF Generation | ReportLab |
| DOCX Generation | python-docx |

### Frontend
| Layer | Technology |
|-------|-----------|
| Framework | Gradio 6.10 |
| Styling | Custom CSS (400+ lines) + Dark Theme |
| Fonts | Inter + Amiri (Urdu support) |
| Deployment | Hugging Face Spaces |

---

## 🏗 Architecture

```
                    ┌─────────────────────────────────┐
                    │         Gradio Frontend          │
                    │  (Chat / Docs / Letter / Loc)   │
                    └──────────────┬──────────────────┘
                                   │ Python API
                    ┌──────────────▼──────────────────┐
                    │         Python Backend          │
                    │  ┌─────────────────────────┐   │
                    │  │  Multi-Turn Chat Engine │   │
                    │  │  Document Analysis      │   │
                    │  │  Legal Letter Generator │   │
                    │  │  Location Help Module   │   │
                    │  └───────────┬─────────────┘   │
                    └──────────────┼─────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │         RAG Retrieval Layer      │
                    │  ┌────────────┐  ┌──────────┐   │
                    │  │   Cohere   │──│ Pinecone │   │
                    │  │ Embeddings │  │ VectorDB │   │
                    │  └────────────┘  └──────────┘   │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │         LLM Inference (Groq)      │
                    │  Llama 3.1-8B → 3.3-70B → Gemma │
                    │  (Auto-fallback on failure)       │
                    └───────────────────────────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               ▼                   ▼                   ▼
         ┌──────────┐      ┌──────────┐      ┌──────────────┐
         │ Whisper  │      │   gTTS   │      │ ReportLab    │
         │  (STT)   │      │  (TTS)   │      │ + python-docx│
         └──────────┘      └──────────┘      └──────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Git
- Tesseract OCR (system dependency)

### 1. Clone the Repository

```bash
git clone https://github.com/Shahrukh-aidev/Advocate-AI.git
cd Advocate-AI
```

### 2. Install Dependencies

```bash
pip install gradio requests pinecone-client cohere reportlab python-docx pdfplumber Pillow pytesseract pdf2image gTTS
```

> **Note**: Install Tesseract OCR on your system:
> - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-urd`
> - **macOS**: `brew install tesseract`
> - **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

### 3. Set Up Environment Variables

Create a `.env` file or export these in your terminal:

```bash
export GROQ_KEY="your_groq_api_key_here"
export PINECONE_KEY="your_pinecone_api_key_here"
export COHERE_KEY="your_cohere_api_key_here"
```

| Variable | Required | Source |
|----------|----------|--------|
| `GROQ_KEY` | ✅ Yes | [groq.com](https://console.groq.com) — Free tier available |
| `PINECONE_KEY` | ✅ Yes | [pinecone.io](https://www.pinecone.io) — Free tier available |
| `COHERE_KEY` | ✅ Yes | [cohere.com](https://cohere.com) — Free tier available |

### 4. Prepare Pinecone Index

Create a Pinecone index named `haq-law` and populate it with Pakistani law embeddings using Cohere's `embed-english-light-v3.0` model.

### 5. Run Locally

```bash
python app.py
```

The app will be available at `http://localhost:7860`

### 6. Deploy to Hugging Face Spaces

```bash
pip install huggingface-hub
huggingface-cli login
# Follow HF Spaces documentation for Gradio SDK deployment
```

---

## 📸 Screenshots

### 💬 Multi-Turn Chat

| Chat Interface | Follow-up Question |
|---|---|
| ![chat](screenshots/chat-interface.jpg) | ![followup](screenshots/followup-chat.jpg) |

### 📄 Document Analysis

| Upload Document | AI Analysis Result |
|---|---|
| ![upload](screenshots/doc-upload.jpg) | ![analysis](screenshots/doc-analysis.jpg) |

### ⚖️ Legal Notice Generator

| Letter Form | PDF Download |
|---|---|
| ![form](screenshots/letter-form.jpg) | ![pdf](screenshots/pdf-download.jpg) |

### 📍 Location Help

| Auto-Detect | Provincial Laws |
|---|---|
| ![location](screenshots/location-detect.jpg) | ![laws](screenshots/provincial-laws.jpg) |

---

## 📁 Project Structure

```
Advocate-AI/
├── app.py                    # Main application (all features in one file)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore
└── screenshots/              # UI screenshots for README
```

> **Note**: This is a single-file Gradio application. All features — chat, document analysis, letter generation, location help, and voice — are contained in `app.py` for easy deployment on Hugging Face Spaces.

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

```bash
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
```

### Areas for Contribution
- 🌐 Add more provincial laws and city data
- 🗣️ Improve Urdu NLP and TTS quality
- 📄 Expand document types for analysis
- 🧪 Add unit tests and CI/CD
- 📱 Improve mobile responsiveness
- 🌍 Add more languages (Sindhi, Punjabi, Pashto)

---

## 👨‍💻 Author

**Shahrukh Hussain** — AI/ML Developer
- GitHub: [@Shahrukh-aidev](https://github.com/Shahrukh-aidev)
- LinkedIn: [shahrukh-hussain](https://www.linkedin.com/in/shahrukh-hussain/)
- Fiverr: [jsharukh123](https://www.fiverr.com/users/jsharukh123/)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>If you found this useful, please ⭐ the repo — it helps a lot!</p>
  <p><strong>"Apna Haq Jaano"</strong> — Know Your Rights</p>
  <p>Built with ❤️ in Pakistan 🇵🇰</p>
</div>
