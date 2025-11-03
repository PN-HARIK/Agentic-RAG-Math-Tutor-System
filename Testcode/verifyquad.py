from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

# Check if collection exists
exists = client.collection_exists("math_qa")
print(f"Collection 'math_qa' exists: {exists}")

# Get collection info including number of points
if exists:
    info = client.get_collection("math_qa")
    print(f"Collection info: {info}")
