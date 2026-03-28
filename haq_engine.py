from pinecone import Pinecone
import requests

# ✅ YOUR KEYS
PINECONE_KEY = "pcsk_5TTMRy_Fd8QwDTPGV8K2iQeuroonvzAsMKUuptmEWMGTPruH3iRQ8hVxpPEnESCueYVQvF"
GROQ_KEY     = "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"

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("haq-laws")

print("✅ HAQ Engine loaded — using local Ollama embeddings!")

SYSTEM_PROMPT = """You are HAQ, Pakistan's most accurate AI legal assistant.

STRICT RULES:
1. Answer ONLY from the law sections provided to you
2. ALWAYS cite exact: Law Name + Section/Article number
3. If laws provided don't answer — say: "I don't have enough legal data on this topic yet."
4. Never guess. Never hallucinate citations.
5. If question is in Urdu — answer in Urdu
6. If question is in English — answer in English
7. Always mention if law is Sindh/Punjab/Federal
8. Keep language simple — user may not be educated

FORMAT:
📖 LEGAL BASIS
[Exact law name + section + text from provided context]

⚖️ THE RULING
[Clear direct answer]

✅ WHAT YOU SHOULD DO
[Step by step practical advice]

⚠️ DISCLAIMER
This is general legal information. For court cases, consult a licensed lawyer (Vakeel).
"""

def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text[:500]
        }
    )
    result = response.json()
    if 'embedding' not in result:
        raise Exception(f"Ollama error: {result}")
    return result['embedding']

def search_laws(question, top_k=5):
    embedding = get_embedding(question)
    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    law_sections = []
    for match in results.matches:
        law_sections.append({
            'law': match.metadata.get('law_name', 'Unknown'),
            'text': match.metadata.get('text', ''),
            'score': match.score
        })
    return law_sections

def ask_haq(question):
    law_sections = search_laws(question)

    context = "RELEVANT PAKISTANI LAW SECTIONS:\n\n"
    for i, section in enumerate(law_sections):
        context += f"[{i+1}] From: {section['law']}\n"
        context += f"{section['text']}\n\n"

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.1,
            "max_tokens": 1000,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{context}\n\nUSER QUESTION: {question}"}
            ]
        }
    )

    data = response.json()

    if 'choices' not in data:
        print("Groq error:", data)
        return "Error getting answer. Please try again."

    return data['choices'][0]['message']['content']

# ✅ TEST
if __name__ == "__main__":
    print("HAQ Legal AI — Testing...\n")
    question = "Can police arrest someone without a warrant in Pakistan?"
    print(f"Question: {question}\n")
    answer = ask_haq(question)
    print(f"Answer:\n{answer}")
