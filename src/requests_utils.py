import json
import requests

url = "http://localhost:1234/v1/embeddings"

def create_embedding(string: str) -> list[float]:
    """
    For testing, you can use the following curl command to create an embedding for a sample string:
    curl http://localhost:1234/v1/embeddings \
      -H "Content-Type: application/json" \
      -d '{
        "model": "text-embedding-nomic-embed-text-v1.5",
        "input": "This is my first test chunk."
      }'
    """

    url = "http://localhost:1234/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "text-embedding-nomic-embed-text-v1.5",
        "input": string
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        embedding = response.json().get("data")[0].get("embedding", [])
        return embedding
    else:
        raise Exception(f"Failed to create embedding: {response.status_code} - {response.text}")