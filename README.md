<div align="center">

<h1>⚖️ HAQ – Pakistan Legal AI</h1>

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

**HAQ** (meaning "Right" in Urdu) is Pakistan's first **RAG-based AI legal assistant** designed to democratize access to justice. It empowers everyday citizens — especially those who cannot afford lawyers — with accurate, actionable legal information in **Urdu and English**.

From asking a simple question to receiving a structured legal answer with exact law citations, uploading a court document for AI analysis, or generating a professionally formatted legal notice with official verification links — the entire pipeline is **automated, intelligent, and completely free**.

> Built by [Shahrukh Hussain](https://github.com/Shahrukh-aidev) — AI/ML Developer & CS Student at Sukkur IBA University 🇵🇰

---

## ✨ Features

### 💬 Multi‑Turn Legal Chat
- **Contextual Memory** — HAQ remembers your full conversation across 10+ turns. Ask a question, get an answer, then follow up with *"What if he doesn't comply?"* or *"Phir kya hoga?"* — no need to repeat yourself.
- **Bilingual Support** — Ask in Urdu or English; HAQ matches your language with pure Urdu (no Hindi) for legal accuracy.
- **Structured Answers** — Every response follows a consistent format: Legal Basis → The Ruling → What You Should Do → Where to Go → Disclaimer.

### 🎙️ Voice Input & 🔊 Voice Output
- **Voice Input** — Speak your legal question in Urdu or English; HAQ transcribes it via Groq Whisper (large‑v3) with high accuracy.
- **Voice Output** — Listen to HAQ's answers in your own language using Google TTS — perfect for low‑literacy users or when you're on the go.

### 📄 Document Upload & Analysis
- Upload any legal document: **court notices, FIR copies, contracts, property deeds, rent agreements, bank documents, or legal notices**.
- **Smart Extraction** — Uses `pdfplumber` for native PDF text extraction (digital PDFs) and **Tesseract OCR** for scanned images and image‑based PDFs.
- **AI Analysis** — HAQ explains what the document means, highlights critical clauses, identifies deadlines, explains legal implications, suggests next steps, and flags red flags.

### ⚖️ Legal Notice Generator
- Generate a **professional Pakistani legal notice** with exact law citations, reference number, and 15‑day standard deadline.
- **Download as PDF** (print‑ready) or **Word (.docx)** (editable) — saves ₨5,000–10,000 in lawyer fees.
- Includes **acknowledgement of receipt** section for legal proof.

### 📍 Location‑Based Legal Help
- **Auto‑detect** your city or select manually from 21+ major Pakistani cities.
- Shows: nearest **High Court**, **District Courts**, **Police Stations**, **Free Legal Aid Centres**, and **local Bar Association contacts**.
- **Provincial Law Variations** — explains how laws differ across Sindh, Punjab, KPK, Balochistan, ICT, AJK, and Gilgit‑Baltistan.

### 🔗 Verification Links
- Every answer includes official links to **pakistancode.gov.pk** so you can verify the law yourself.
- Covers: Constitution 1973, PPC 1860, CrPC 1898, MFLO 1961, PECA 2016, Contract Act 1872, TPA 1882, Labour Laws, and more.

---

## 🛠 Tech Stack

### Backend
| Layer | Technology |
|-------|-----------|
| Framework | Python + Gradio |
| LLM | Groq (Llama 3.1‑8B / 70B, Gemma 2) |
| Embeddings | Cohere (embed‑english‑light‑v3.0) |
| Vector DB | Pinecone (4000+ laws indexed) |
| Speech‑to‑Text | Groq Whisper (large‑v3) |
| Text‑to‑Speech | gTTS (Google TTS) |
| OCR | Tesseract + pdf2image + pdfplumber |
| PDF Generation | ReportLab |
| DOCX Generation | python‑docx |

### Frontend
| Layer | Technology |
|-------|-----------|
| Framework | Gradio 6.10 |
| Styling | Custom CSS + Dark Mode |
| Deployment | Hugging Face Spaces |

---

## 🏗 Architecture
