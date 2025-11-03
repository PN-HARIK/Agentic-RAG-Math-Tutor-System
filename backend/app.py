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
    """Retrieve from knowledge base with threshold filtering."""
    query_embedding = embedding_model.encode(user_query).tolist()
    results = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    hits = results[0] if isinstance(results, tuple) else results
    context_chunks = []
    for hit in hits:
        if hasattr(hit, "payload") and isinstance(hit.payload, dict):
            score = getattr(hit, "score", None)
            q = hit.payload.get("Question")
            a = hit.payload.get("Answer")
            if q and a and (score is None or score >= relevance_threshold):
                context_chunks.append(f"Question: {q}\nAnswer: {a}")

    return "\n\n".join(context_chunks) if context_chunks else None


def web_search(query: str) -> str:
    """Fallback web search if no KB here."""
    print(f"Web search fallback triggered for: {query}")
    return (
        "Web search returned related math concepts:\n"
        "- Basic math principles.\n- Guide for problem solving steps."
    )


def generate_answer(user_query, context):
    prompt = f"""
You are a math tutoring AI developed with strong guardrails:
- Use only math/educational content from trusted KB or web search.
- Quote context exactly.
- If no context, say 'No relevant context — solving from first principles.'
- Show reasoning and final succinct answer.

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
    if not context:
        context = web_search(user_query)

    answer = generate_answer(user_query, context)

    return {"answer": answer, "context": context}
