from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

def ask_llm(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)

    response = client.chat.completions.create(
        model="your-chat-model-name",
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions only using the provided context. "
                    "If the answer is not in the context, say that you don't know."
                    "Answer in language of origin of the question."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content