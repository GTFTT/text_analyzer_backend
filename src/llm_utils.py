from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

def ask_llm(question: str, chunks: list[str], chat_history: list[ChatCompletionMessageParam] | None = None) -> str:
    context = "\n\n---\n\n".join(chunks)

    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": (
            "You answer questions only using the provided context and the conversation history. "
            "If the answer is not in the context or history, say that you don't know. "
            "Answer in language of origin of the question."
        )
    }

    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{question}"
    }

    messages: list[ChatCompletionMessageParam] = [
        system_message
    ]

    if chat_history:
        messages.extend(chat_history)

    messages.append(user_message)

    response = client.chat.completions.create(
        model="gemma-3-4b-it-qat",
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content