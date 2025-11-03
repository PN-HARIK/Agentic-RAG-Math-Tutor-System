from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "math_qa"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

qdrant = QdrantClient(QDRANT_URL)
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


user_query = input("Enter a math question to test relevance: ")


query_vector = model.encode(user_query).tolist()


results = qdrant.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=5,
    with_payload=True,
)

print(f"\n🔍 Top matches for query: '{user_query}'")
for hit in results:
    payload = hit.payload
    score = hit.score

    question = payload.get("source_question") or payload.get("Question")
    answer = payload.get("source_answer") or payload.get("Answer")
    text = payload.get("text")

    print(f"\nScore: {score:.4f}")
    if question and answer:
        print(f"Question: {question}\nAnswer: {answer}")
    elif text:
        print(f"Text: {text}")
    else:
        print("⚠️ No recognizable fields found.")
