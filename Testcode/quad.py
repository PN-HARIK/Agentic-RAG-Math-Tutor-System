from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize model and Qdrant
model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient("http://localhost:6333")

collection_name = "math_qa"

while True:
    query = input("Enter your question (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    query_vector = model.encode(query).tolist()

    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )

    print("\nRetrieved Context:")
    if not search_results:
        print("(No relevant context found)\n")
    else:
        for i, result in enumerate(search_results, 1):
            print(f"{i}. {result.payload.get('text', '(No text)')}")

    print("\n" + "-" * 80 + "\n")
