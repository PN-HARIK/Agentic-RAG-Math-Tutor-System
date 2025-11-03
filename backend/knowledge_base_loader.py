import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams


def load_knowledge_base(folder_path="D:/MathAI/knowledge_base"):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Knowledge base folder not found: {folder_path}")
    knowledge_base = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    knowledge_base.extend(data)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping {filename}: Invalid JSON ({e})")
    print(f"✅ Loaded {len(knowledge_base)} Q&A items from {folder_path}")
    return knowledge_base



def recursive_character_text_splitter(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    text_length = len(text)
    start = 0
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks



def main():
    knowledge_base = load_knowledge_base()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    qdrant_client = QdrantClient("http://localhost:6333")
    collection_name = "math_qa"
    vector_size = 384  

    
    if qdrant_client.collection_exists(collection_name):
        print("🗑️  Deleting existing collection...")
        qdrant_client.delete_collection(collection_name)

    
    print("🧠 Creating collection in Qdrant...")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance="Cosine")
    )


    print("🚀 Generating embeddings and preparing points...")
    points = []
    point_id = 0
    batch_size = 100  

    total = len(knowledge_base)
    for idx, item in enumerate(knowledge_base):
        if idx % 50 == 0:
            print(f"Processing {idx}/{total}...")

        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        combined_text = f"Question: {question}\nAnswer: {answer}"

        chunks = recursive_character_text_splitter(combined_text)
        embeddings = model.encode(chunks, convert_to_numpy=True)

        for chunk_text, embedding in zip(chunks, embeddings):
            embedding = embedding.astype(float).tolist()  
            payload = {
                "text": chunk_text,
                "source_question": question,
                "source_answer": answer
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))
            point_id += 1

            
            if len(points) >= batch_size:
                try:
                    qdrant_client.upsert(collection_name=collection_name, points=points)
                    points.clear()
                except Exception as e:
                    print("❌ Upsert failed:", e)
                    points.clear()


    if points:
        qdrant_client.upsert(collection_name=collection_name, points=points)

    print(f"✅ Ingested {point_id} chunks into Qdrant collection '{collection_name}' successfully.")


if __name__ == "__main__":
    main()
