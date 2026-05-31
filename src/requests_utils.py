import json
import requests

url = "http://localhost:1234/v1/embeddings"


class EmbeddingServiceError(Exception):
    """Raised when the embedding service cannot provide a valid embedding."""

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

    headers = {"Content-Type": "application/json"}
    data = {
        "model": "text-embedding-nomic-embed-text-v1.5",
        "input": string
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
    except requests.RequestException as exc:
        raise EmbeddingServiceError(
            f"Embedding service is unavailable at {url}."
        ) from exc

    if response.status_code != 200:
        raise EmbeddingServiceError(
            f"Embedding service returned {response.status_code}: {response.text}"
        )

    payload = response.json()
    items = payload.get("data") if isinstance(payload, dict) else None
    if not items or not isinstance(items, list):
        raise EmbeddingServiceError("Embedding service returned an invalid response payload.")

    embedding = items[0].get("embedding", [])
    if not isinstance(embedding, list):
        raise EmbeddingServiceError("Embedding service did not return a valid embedding vector.")

    return embedding
