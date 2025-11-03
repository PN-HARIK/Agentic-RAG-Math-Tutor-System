from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import google.generativeai as genai
import os


API_KEY = os.getenv("GENAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing GENAI_API_KEY environment variable")

genai.configure(api_key=API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant_client = QdrantClient("http://localhost:6333")
collection_name = "math_qa"
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


class Query(BaseModel):
    text: str


def retrieve_context(user_query, top_k=5, relevance_threshold=0.6):
    query_embedding = embedding_model.encode(user_query).tolist()

    try:
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )
    except Exception as e:
        print(f"Qdrant search error: {e}")
        return None

    print(f"Search returned {len(results)} results") 

    context_chunks = []
    for hit in results:
        score = hit.score if hasattr(hit, "score") else None
        payload = hit.payload or {}
        question = payload.get("source_question") or payload.get("Question")
        answer = payload.get("source_answer") or payload.get("Answer")
        text = payload.get("text")

        print(f"Hit score: {score}, question: {question}, answer: {answer}, text: {text}") 

        if question and answer and (score is None or score >= relevance_threshold):
            context_chunks.append(f"Question: {question}\nAnswer: {answer}")
        elif text and (score is None or score >= relevance_threshold):
            context_chunks.append(f"Text: {text}")

    if not context_chunks:
        print("⚠️ No relevant context found in KB matching threshold.")
        return None

    return "\n\n".join(context_chunks)


def generate_answer(user_query, context):
    prompt = f"""
You are a reasoning math LLM assistant.
Follow guardrails:

1. Use only educational math content from your knowledge base.
2. Quote context facts exactly if available.
3. If no reliable context, clearly say 'No relevant context — solving from first principles.'
4. Show step-by-step workings and final numeric answer.

Context:
{context or '(No relevant context)'}

Question:
{user_query}

Answer:
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if hasattr(response, "text") else "(No response)"
    except Exception as e:
        return f"(Error: {str(e)})"


@app.post("/ask")
async def ask_question(query: Query):
    user_query = query.text.strip()
    if not user_query:
        return {"answer": "Please enter a question."}

   
    context = retrieve_context(user_query)

   
    answer = generate_answer(user_query, context)

    return {"answer": answer, "context": context}
