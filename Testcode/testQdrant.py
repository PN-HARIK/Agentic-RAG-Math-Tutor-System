from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")

print(client.collection_exists("math_qa"))
