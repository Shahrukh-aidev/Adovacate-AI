<div align="center">
<img src="https://img.shields.io/badge/⚖️-HAQ-FFD700?style=for-the-badge&logoColor=white" alt="HAQ Logo" width="120">
⚖️ HAQ – Pakistan Legal AI
Pakistan's first RAG-based AI legal assistant — Multi-turn chat, document analysis, legal notice generation & location-based legal help for citizens who cannot afford lawyers.
<p>
  <a href="https://github.com/Shahrukh-aidev/Advocate-AI/stargazers"><img src="https://img.shields.io/github/stars/Shahrukh-aidev/Advocate-AI?style=for-the-badge&color=FFD700" alt="Stars"/></a>
  <a href="https://github.com/Shahrukh-aidev/Advocate-AI/network/members"><img src="https://img.shields.io/github/forks/Shahrukh-aidev/Advocate-AI?style=for-the-badge&color=4FC3F7" alt="Forks"/></a>
  <a href="https://github.com/Shahrukh-aidev/Advocate-AI/issues"><img src="https://img.shields.io/github/issues/Shahrukh-aidev/Advocate-AI?style=for-the-badge&color=FF7043" alt="Issues"/></a>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/AI-Groq%20Powered-blueviolet?style=for-the-badge" alt="Groq"/>
  <img src="https://img.shields.io/badge/RAG-Pinecone%20%2B%20Cohere-orange?style=for-the-badge" alt="RAG"/>
  <img src="https://img.shields.io/badge/Deployed-Hugging%20Face%20Spaces-blue?style=for-the-badge" alt="Spaces"/>
  <img src="https://img.shields.io/badge/Languages-Urdu%20%7C%20English-10B981?style=for-the-badge" alt="Languages"/>
</p>
<p>
  <a href="#-what-is-haq">About</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-live-demo">Demo</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-project-structure">Structure</a> •
  <a href="#-contributing">Contributing</a>
</p>
</div>
🧠 What is HAQ?
HAQ (حق — meaning "Right" in Urdu) is Pakistan's first RAG-based AI legal assistant designed to democratize access to justice. It empowers everyday citizens — especially those who cannot afford lawyers — with accurate, actionable legal information in Urdu and English.
From asking a simple legal question to receiving a structured answer with exact law citations, uploading a court document for AI analysis, or generating a professionally formatted legal notice with official verification links — the entire pipeline is automated, intelligent, and completely free.
🎯 Mission: Because 90% of Pakistanis cannot afford a lawyer.
Built with ❤️ by Shahrukh Hussain — AI/ML Developer & CS Student at Sukkur IBA University 🇵🇰
✨ Key Features
💬 Multi-Turn Legal Chat with Memory
Contextual Memory — HAQ remembers your full conversation across 10+ turns. Ask a question, get an answer, then follow up with "What if he doesn't comply?" or "Phir kya hoga?" — no need to repeat yourself.
Bilingual Support — Ask in Urdu or English; HAQ matches your language with pure Urdu (no Hindi words) for legal accuracy.
Structured Answers — Every response follows a consistent format:
LEGAL BASIS → THE RULING → WHAT YOU SHOULD DO → WHERE TO GO → DISCLAIMER
Safety Guardrails — Built-in filters reject harmful, illegal, or jailbreak prompts.
🎙️ Voice Input & 🔊 Voice Output
Voice Input — Speak your legal question in Urdu or English; HAQ transcribes it via Groq Whisper (large-v3) with high accuracy.
Voice Output — Listen to HAQ's answers in your own language using Google TTS (gTTS) — perfect for low-literacy users or when you're on the go.
Auto Language Detection — Automatically detects Urdu vs English for TTS playback.
📄 Smart Document Upload & Analysis
Upload any legal document: Court notices, FIR copies, contracts, property deeds, rent agreements, bank documents, divorce papers, or legal notices.
Dual Extraction Engine:
pdfplumber for native text extraction from digital PDFs (fast & accurate)
Tesseract OCR for scanned images and image-based PDFs (supports English + Urdu)
Comprehensive AI Analysis:
📄 Document Overview
⚠️ Critical Clauses / Warnings
📋 Key Dates & Deadlines
⚖️ Legal Implications
✅ What You Should Do Next
🚨 Red Flags
📞 Where to Get Help
⚖️ Professional Legal Notice Generator
Generate a professional Pakistani legal notice with exact law citations, reference number (LN/YEAR/UNIQUE), and a 15-day standard deadline.
Party Structure: FROM (Murasil / Noticee) → TO (Mukhatib / Respondent)
Download as PDF (print-ready with formal styling) or Word (.docx) (editable) — saves ₨5,000–10,000 in lawyer fees.
Includes Acknowledgement of Receipt section for legal proof.
Auto Verification Links — Detects laws cited in the notice and generates official pakistancode.gov.pk verification cards.
📍 Location-Based Legal Help (21+ Cities)
Auto-detect your city via browser geolocation or select manually.
Covered Cities: Karachi, Lahore, Islamabad, Peshawar, Quetta, Multan, Faisalabad, Rawalpindi, Gujranwala, Sargodha, Bahawalpur, Hyderabad, Sukkur, Larkana, Mardan, Swat/Mingora, Abbottabad, Gwadar, Mirpur, Muzaffarabad, Gilgit, and more.
Local Resources:
🏛 High Court & District Courts
🚔 Police Stations & HQ
⚖ Free Legal Aid Centres (with phone numbers)
📞 Local Bar Association Contacts
Provincial Law Variations — Explains how laws differ across:
Sindh vs Punjab vs KPK vs Balochistan vs ICT vs AJK vs Gilgit-Baltistan
Covers: Tenancy, Labour, Family, Cybercrime, Local Government, Women Protection, RTI
🔗 Official Law Verification
Every answer includes clickable verification cards linking directly to pakistancode.gov.pk (Pakistan Ministry of Law & Justice).
14 Major Laws auto-detected: Constitution 1973, PPC 1860, CrPC 1898, PECA 2016, MFLO 1961, Contract Act 1872, TPA 1882, Labour Laws, Registration Act 1908, Specific Relief Act 1877, Anti-Terrorism Act 1997, DMMA 1939, Qanoon-e-Shahadat 1984, Banking Ordinance 1962.
🎨 Premium Dark UI
Custom-designed dark theme with gold accents inspired by Pakistani legal heritage.
Fully responsive for mobile and desktop.
Sticky navigation, smooth animations, and accessibility-focused design.
🚀 Live Demo
https://huggingface.co/spaces/Shahrukh350/Advocate-AI
Try it now: HAQ on Hugging Face Spaces
🛠 Tech Stack
Backend
Table
Layer	Technology	Purpose
Framework	Python + Gradio 6.10	UI & API layer
LLM	Groq API (Llama 3.1-8B / 3.3-70B, Gemma 2-9B)	Fast inference with fallback chain
Embeddings	Cohere (embed-english-light-v3.0)	Semantic search vectors
Vector DB	Pinecone (haq-law index)	4000+ laws indexed for RAG
Speech-to-Text	Groq Whisper (large-v3)	Urdu & English voice transcription
Text-to-Speech	gTTS (Google TTS)	Voice answers in Urdu/English
OCR	Tesseract + pdf2image + pdfplumber	Document text extraction
PDF Generation	ReportLab	Print-ready legal notices
DOCX Generation	python-docx	Editable legal notices
Frontend
Table
Layer	Technology
Framework	Gradio 6.10
Styling	Custom CSS (400+ lines) + Dark Theme
Fonts	Inter + Amiri (Urdu support)
Deployment	Hugging Face Spaces
🏗 Architecture
plain
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  💬 Chat  │ │ 📄 Docs  │ │ ⚖ Letter│ │ 📍 Loc    │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       │            │            │            │                │
│       ▼            ▼            ▼            ▼                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              GRADIO BACKEND (Python)                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │ Multi-Turn  │  │   Document  │  │   Legal     │     │   │
│  │  │   Chat      │  │   Analysis  │  │   Letter    │     │   │
│  │  │   Engine    │  │   Pipeline  │  │  Generator  │     │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │   │
│  │         │                │                │              │   │
│  │         ▼                ▼                ▼              │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │              RAG RETRIEVAL LAYER                 │   │   │
│  │  │  ┌────────────┐      ┌─────────────────────┐   │   │   │
│  │  │  │  Cohere    │─────▶│      Pinecone       │   │   │   │
│  │  │  │ Embeddings │      │   Vector Database   │   │   │   │
│  │  │  └────────────┘      │   (4000+ Laws)      │   │   │   │
│  │  │                      └─────────────────────┘   │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  │                         │                             │   │
│  │                         ▼                             │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │              LLM INFERENCE (Groq)                │   │   │
│  │  │  Llama 3.1-8B → Llama 3.3-70B → Gemma 2-9B     │   │   │
│  │  │  (Automatic fallback chain on failure/timeout)   │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │           │           │                               │
│       ▼           ▼           ▼                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │ Whisper │ │  gTTS   │ │ReportLab│                        │
│  │  (STT)  │ │  (TTS)  │ │+ python │                        │
│  │         │ │         │ │ -docx   │                        │
│  └─────────┘ └─────────┘ └─────────┘                        │
└─────────────────────────────────────────────────────────────────┘
📦 Getting Started
Prerequisites
Python 3.9+
Git
1. Clone the Repository
bash
git clone https://github.com/Shahrukh-aidev/Advocate-AI.git
cd Advocate-AI
2. Install Dependencies
bash
pip install gradio requests pinecone-client cohere reportlab python-docx pdfplumber Pillow pytesseract pdf2image gTTS
Note: For OCR support, you also need Tesseract OCR installed on your system:
Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-urd
macOS: brew install tesseract
Windows: Download from UB Mannheim
3. Set Up Environment Variables
Create a .env file or set these in your environment / Hugging Face Space Secrets:
bash
export GROQ_KEY="your_groq_api_key_here"
export PINECONE_KEY="your_pinecone_api_key_here"
export COHERE_KEY="your_cohere_api_key_here"
Table
Variable	Required	Source
GROQ_KEY	✅ Yes	groq.com — Free tier available
PINECONE_KEY	✅ Yes	pinecone.io — Free tier available
COHERE_KEY	✅ Yes	cohere.com — Free tier available
4. Prepare Pinecone Index
Create a Pinecone index named haq-law and populate it with Pakistani law embeddings using Cohere's embed-english-light-v3.0 model.
5. Run Locally
bash
python app.py
The app will be available at http://localhost:7860
6. Deploy to Hugging Face Spaces
bash
# Install Hugging Face CLI
pip install huggingface-hub

# Login
huggingface-cli login

# Create and push Space
# (Follow HF Spaces documentation for Gradio SDK)
📁 Project Structure
plain
Advocate-AI/
├── app.py                    # Main application (all features)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore
└── assets/
    ├── screenshots/          # UI screenshots
    └── demo/                 # Demo videos/GIFs
Note: This is a single-file Gradio application. All features — chat, document analysis, letter generation, location help, and voice — are contained in app.py for easy deployment on Hugging Face Spaces.
💡 Usage Examples
Example 1: Multi-Turn Chat
plain
You: Police ne mujhe bina warrant arrest kiya. Mera kya haq hai?
HAQ: [Explains Article 10, CrPC Section 54, 24-hour rule...]

You: What if they refuse to release me?
HAQ: [Remembers context → explains Habeas Corpus, Article 199, High Court writ...]

You: Phir kya hoga?
HAQ: [Urdu follow-up → Urdu answer, same case context...]
Example 2: Document Analysis
Upload a scanned court notice (PDF or image)
Select "Court Notice / Summons" as document type
HAQ extracts text via OCR and provides:
Document summary
Critical deadlines
Legal implications
Next steps
Example 3: Legal Notice Generation
Fill: Your name, Other party name, Your address
Select: "Legal Notice to Landlord for Illegal Eviction"
Describe your situation with dates and facts
HAQ generates a professional notice with:
Exact law citations (e.g., Sindh Tenancy Act 1950)
15-day compliance deadline
PDF + DOCX download
🗺️ Supported Locations & Provincial Laws
Table
Province	Cities Covered	Key Local Variations
Sindh	Karachi, Hyderabad, Sukkur, Larkana	Tenancy Act 1950, Domestic Violence Act 2013
Punjab	Lahore, Rawalpindi, Faisalabad, Multan, Gujranwala, Sargodha, Bahawalpur	Rented Premises Act 2009, Women Protection Act 2016
KPK	Peshawar, Mardan, Swat/Mingora, Abbottabad	Domestic Violence Act 2021, RTI Act 2013
Balochistan	Quetta, Gwadar	Tenancy Act 1948, Local Government Act 2010
ICT	Islamabad	Rent Restriction Ordinance 2001, Domestic Violence Act 2012
AJK	Mirpur, Muzaffarabad	AJK-specific family & labour courts
Gilgit-Baltistan	Gilgit	GB Governance Order 2018
🤝 Contributing
Contributions are welcome! Please read our Contributing Guidelines first.
Fork the repository
Create your feature branch: git checkout -b feature/amazing-feature
Commit your changes: git commit -m 'Add amazing feature'
Push to the branch: git push origin feature/amazing-feature
Open a Pull Request
Areas for Contribution
🌐 Add more provincial laws and city data
🗣️ Improve Urdu NLP and TTS quality
📄 Expand document types for analysis
🧪 Add unit tests and CI/CD
📱 Improve mobile responsiveness
🌍 Add more languages (Sindhi, Punjabi, Pashto)
📜 License
This project is licensed under the MIT License — see the LICENSE file for details.
🙏 Acknowledgements
Groq for blazing-fast LLM inference
Pinecone for vector search infrastructure
Cohere for embeddings
Gradio for the wonderful UI framework
Hugging Face for free deployment platform
Pakistan Ministry of Law & Justice for pakistancode.gov.pk
All open-source contributors who make projects like this possible
📬 Contact
<p>
  <a href="https://github.com/Shahrukh-aidev"><img src="https://img.shields.io/badge/GitHub-Shahrukh--aidev-181717?style=for-the-badge&logo=github"/></a>
</p>
Developer: Shahrukh Hussain
Location: Sukkur, Sindh, Pakistan 🇵🇰
Institution: Sukkur IBA University
"Apna Haq Jaano" — Know Your Rights
<div align="center">
  <img src="https://img.shields.io/badge/Made%20with%20❤️%20in-Pakistan-00A651?style=for-the-badge" alt="Made in Pakistan"/>
  <br><br>
  <sub>If this project helped you, please ⭐ star the repository!</sub>
</div>
