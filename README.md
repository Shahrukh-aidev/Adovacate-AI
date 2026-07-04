<div align="center">

<img src="https://img.shields.io/badge/%E2%9A%96%EF%B8%8F-HAQ-1a1a2e?style=for-the-badge&labelColor=1a1a2e&color=4f46e5" alt="HAQ"/>

<h1>HAQ — حق<br/><sub>Pakistan's AI-Powered Legal Assistant</sub></h1>

<p>
  <strong>Bilingual RAG pipeline · Document OCR & Analysis · Voice I/O · Auto-generated Legal Letters</strong><br/>
  Plain-language legal guidance in English & Urdu — grounded in 100+ Pakistani Acts, zero hallucination.
</p>

<p>
  <a href="https://github.com/Shahrukh-aidev/HAQ/stargazers">
    <img src="https://img.shields.io/github/stars/Shahrukh-aidev/HAQ?style=for-the-badge&color=FFD700" alt="Stars"/>
  </a>
  <a href="https://github.com/Shahrukh-aidev/HAQ/network/members">
    <img src="https://img.shields.io/github/forks/Shahrukh-aidev/HAQ?style=for-the-badge&color=4FC3F7" alt="Forks"/>
  </a>
  <a href="https://github.com/Shahrukh-aidev/HAQ/issues">
    <img src="https://img.shields.io/github/issues/Shahrukh-aidev/HAQ?style=for-the-badge&color=FF7043" alt="Issues"/>
  </a>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/LLM-Groq%20%2B%20Llama-7c3aed?style=for-the-badge" alt="Groq"/>
  <img src="https://img.shields.io/badge/Deploy-HuggingFace%20Spaces-f97316?style=for-the-badge" alt="HuggingFace"/>
</p>

<p>
  <a href="#-what-is-haq">About</a> ·
  <a href="#-key-features">Features</a> ·
  <a href="#-how-it-works">How It Works</a> ·
  <a href="#-tech-stack">Tech Stack</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-getting-started">Getting Started</a> ·
  <a href="#-supported-laws">Supported Laws</a> ·
  <a href="#-contributing">Contributing</a>
</p>

</div>

---

## 🧠 What is HAQ?

**HAQ** (حق — Urdu for *"right"* or *"justice"*) is an open-source bilingual AI legal assistant built for the 220 million citizens of Pakistan who face real legal problems but lack access to affordable legal counsel.

HAQ answers legal questions in plain English or Urdu, analyzes uploaded documents (FIRs, court notices, contracts), generates formal legal letters, and speaks responses aloud — all grounded in a RAG pipeline over **100+ Pakistani Acts and 4,000+ indexed legal chunks**, with every answer citing specific sections and verified links to `pakistancode.gov.pk`.

> Built by [Shahrukh Baloch](https://github.com/Shahrukh-aidev) — AI/ML Developer & CS Student at Sukkur IBA University 🇵🇰

---

## ✨ Key Features

### ⚖️ Cited Legal Q&A
| Capability | Details |
|---|---|
| RAG Retrieval | Cohere embeddings + Pinecone vector search across 100+ Acts before every answer |
| Anti-Hallucination | Every response includes section numbers + verified `pakistancode.gov.pk` links |
| 4-Model Fallback | Groq-hosted Llama models cascade automatically — near 100% uptime |
| Bilingual | Auto-detects language; replies in English or Urdu to match the question |
| Location-Aware | Routes users to the correct court, police station, or government office by city/province |

### 📄 Document Analysis
| Capability | Details |
|---|---|
| OCR Engine | Tesseract (`eng+urd`) with contrast boost, upscaling, and multi-PSM strategy for poor scans |
| PDF Extraction | Native layout-preserving text via `pdfplumber`; OCR fallback for fully scanned PDFs |
| Smart Analysis | Structured output: overview → critical clauses → key dates → legal implications → red flags → next steps |
| Red Flag Detection | Flags suspicious, one-sided, or legally risky clauses automatically |
| Supported Formats | PDF, PNG, JPG, TIFF, WEBP, TXT |

### 🔊 Voice Interface
| Capability | Details |
|---|---|
| Voice Input | Record in Urdu or English — Groq Whisper Large v3 transcribes in seconds |
| Text-to-Speech | Microsoft edge-tts neural voices: `ur-PK-UzmaNeural` (Urdu), `en-US-JennyNeural` (English) |
| Fallback TTS | gTTS activates automatically if edge-tts is unavailable |

### 📝 Legal Document Generation
| Capability | Details |
|---|---|
| PDF Letters | ReportLab-generated formal letters with proper headers, section citations, and layout |
| DOCX Letters | Word-compatible documents with professional styling via python-docx |
| Law Coverage | 14+ major Pakistani statutes supported with auto-detected references |

### 🔗 Verification Links
Every answer automatically detects which laws are referenced and injects a clickable verification card panel — letting users read the actual statute on `pakistancode.gov.pk` with one tap.

---

## 🔄 How It Works

```
User Input (text / voice / document)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                   Input Processing                   │
│  Voice → Groq Whisper STT                           │
│  Document → pdfplumber / Tesseract OCR              │
│  Text → passed directly                             │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                   RAG Pipeline                       │
│  1. Cohere embeds the query                         │
│  2. Pinecone retrieves top-k law chunks             │
│  3. Context + query sent to Groq LLM                │
│  4. Anti-hallucination layer injects law links      │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                   Output Layer                       │
│  • Cited text answer (section numbers + links)      │
│  • Audio via edge-tts / gTTS                        │
│  • Optional: Legal letter PDF or DOCX               │
└─────────────────────────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Runtime** | Python 3.10+ |
| **UI Framework** | Gradio |
| **LLM** | Groq — Llama (4-model fallback chain) |
| **Embeddings** | Cohere |
| **Vector Database** | Pinecone (`haq-law` index, 4,000+ chunks) |
| **Speech-to-Text** | Groq Whisper Large v3 |
| **Text-to-Speech** | Microsoft edge-tts (neural) + gTTS (fallback) |
| **OCR** | Tesseract `eng+urd` + pytesseract + Pillow preprocessing |
| **PDF Processing** | pdfplumber (native) + pdf2image (OCR fallback) |
| **Document Generation** | ReportLab (PDF) + python-docx (DOCX) |
| **Deployment** | HuggingFace Spaces |

---

## 🏗 Architecture

```
                   ┌───────────────────────────────────┐
                   │            Gradio UI               │
                   │  Chat · Document Upload · Voice    │
                   └─────────────────┬─────────────────┘
                                     │
              ┌──────────────────────┼──────────────────┐
              ▼                      ▼                   ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │  Voice Module    │  │  Document Module │  │   Q&A Module     │
   │  Groq Whisper    │  │  pdfplumber      │  │  Cohere Embed    │
   │  STT → text      │  │  Tesseract OCR   │  │  Pinecone RAG    │
   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
            └──────────────────── ┼ ──────────────────────┘
                                  │
                   ┌──────────────▼─────────────────┐
                   │       Groq LLM Engine           │
                   │  Model 1 → 2 → 3 → 4 fallback  │
                   │  Anti-hallucination prompting   │
                   │  Section citation injection     │
                   └──────────────┬─────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                    ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │   Text Answer    │  │  Audio Output    │  │ Document Output  │
   │  + Law Links     │  │  edge-tts/gTTS   │  │  PDF / DOCX      │
   └──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- API keys: Groq, Pinecone, Cohere
- System packages: `tesseract-ocr`, `tesseract-ocr-urd`, `poppler-utils`, `ffmpeg`

### 1 · Clone

```bash
git clone https://github.com/Shahrukh-aidev/HAQ.git
cd HAQ
```

### 2 · System Dependencies

**On Ubuntu / Debian:**
```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-urd poppler-utils ffmpeg
```

**On HuggingFace Spaces — `packages.txt`:**
```
tesseract-ocr
tesseract-ocr-urd
poppler-utils
ffmpeg
```

### 3 · Python Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
gradio
pinecone-client
cohere
requests
reportlab
python-docx
pytesseract
Pillow
pdf2image
pdfplumber
gtts
edge-tts
```

### 4 · Environment Variables

Create a `.env` file or set as Secrets (HuggingFace Spaces → Settings → Secrets):

```env
PINECONE_KEY=your_pinecone_api_key
COHERE_KEY=your_cohere_api_key
GROQ_KEY=your_groq_api_key
```

### 5 · Run

```bash
python app.py
```

The Gradio UI launches at `http://localhost:7860`.

> **HuggingFace Spaces:** Push the repo with `packages.txt`, `requirements.txt`, and your Secrets set — it deploys automatically.

---

## 📚 Supported Laws

HAQ covers **14 major Pakistani statutes**, each with pattern-matched detection and direct verified links to `pakistancode.gov.pk`:

| # | Statute | Covers |
|---|---------|--------|
| 🏛 | **Constitution of Pakistan 1973** | Fundamental rights, writs, habeas corpus, mandamus |
| ⚖️ | **Pakistan Penal Code 1860** | Crimes, qatl, diyat, theft, cheating, rape |
| 📋 | **Code of Criminal Procedure 1898** | FIR, bail, challan, arrest |
| 💻 | **Prevention of Electronic Crimes Act 2016** | Cybercrime, online harassment, FIA cases |
| 👪 | **Muslim Family Laws Ordinance 1961** | Nikah, talaq, khula, mehr, iddat |
| 📝 | **Contract Act 1872** | Breach of contract, consideration, voidability |
| 🏠 | **Transfer of Property Act 1882** | Property transfer, mortgage, adverse possession |
| 💼 | **Payment of Wages Act 1936** | Wage disputes, gratuity, EOBI, workmen compensation |
| 📄 | **Registration Act 1908** | Sub-registrar, stamp duty, document registration |
| 🔍 | **Specific Relief Act 1877** | Specific performance, injunctions |
| 🚨 | **Anti-Terrorism Act 1997** | ATC jurisdiction, terrorism-related charges |
| 📜 | **Dissolution of Muslim Marriages Act 1939** | Judicial divorce for Muslim women |
| 🔎 | **Qanoon-e-Shahadat Order 1984** | Evidence, confessions, electronic evidence |
| 🏦 | **Banking Companies Ordinance 1962** | Banking disputes, SBP, banking mohtasib |

---

## 📁 Project Structure

```
HAQ/
├── app.py                  # Entire application — all modules in one file
│   ├── OCR / Document Extraction
│   ├── Text-to-Speech (edge-tts + gTTS)
│   ├── Voice Input (Groq Whisper)
│   ├── RAG Pipeline (Cohere + Pinecone + Groq)
│   ├── Law Detection & Verification Links
│   ├── Document Generation (PDF + DOCX)
│   └── Gradio UI
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level dependencies (HF Spaces)
├── data/
│   └── acts/               # Source documents — 100+ Pakistani Acts
└── README.md
```

---

## 🤝 Contributing

Contributions are welcome — bug fixes, new law patterns, UI improvements, or Urdu NLP enhancements.

```bash
# Fork → branch → PR
git checkout -b feature/your-feature-name
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
```

Please open an issue first for significant changes so we can discuss the approach.

---

## 👨‍💻 Author

**Shahrukh Baloch** — AI/ML Developer

[![GitHub](https://img.shields.io/badge/GitHub-Shahrukh--aidev-181717?style=flat-square&logo=github)](https://github.com/Shahrukh-aidev)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-shahrukh--baloch-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/shahrukh-baloch/)
[![Fiverr](https://img.shields.io/badge/Fiverr-jsharukh123-1DBF73?style=flat-square&logo=fiverr)](https://www.fiverr.com/users/jsharukh123/)

---

## 📄 License

Licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>⭐ Star the repo if HAQ helped you — it genuinely matters.</sub><br/>
  <sub>Built with ❤️ for Pakistan 🇵🇰</sub>
</div>
