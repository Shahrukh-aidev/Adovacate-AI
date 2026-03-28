from pinecone import Pinecone
import requests
import os
import time

# ✅ YOUR PINECONE KEY
PINECONE_KEY = "pcsk_6ed6bd_ThFCdih1V1zRtGDSHfobBzuY88bqw8orHFekUK4b3AR8b5nxBk5Vab8k1oFv73i"

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("haq-laws")

print("✅ Connected to Pinecone!")
print("✅ Using Ollama local embeddings — no API limits!")

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

def chunk_text(text, chunk_size=400):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def upload_law_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    chunks = chunk_text(text)
    law_name = os.path.basename(filepath).replace('.txt', '')
    print(f"\nUploading: {law_name} — {len(chunks)} chunks...")

    vectors = []
    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 50:
            continue

        try:
            embedding = get_embedding(chunk)
            chunk_id = f"{law_name}_{i}"

            vectors.append({
                'id': chunk_id,
                'values': embedding,
                'metadata': {
                    'text': chunk,
                    'law_name': law_name,
                    'source': filepath
                }
            })

            if len(vectors) >= 10:
                index.upsert(vectors=vectors)
                print(f"  ✅ Uploaded chunks up to {i}")
                vectors = []

        except Exception as e:
            print(f"  ❌ Error on chunk {i}: {e}")
            continue

    if vectors:
        index.upsert(vectors=vectors)

    print(f"✅ Done: {law_name}")

# ✅ YOUR FOLDER
folder = r"D:\HAQ-LAWS"

for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith('.txt'):
            filepath = os.path.join(root, file)
            upload_law_file(filepath)

print("\n🎉 All laws uploaded to Pinecone successfully!")
print("No API limits. No costs. Runs forever!")
