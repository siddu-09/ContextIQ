SYSTEM_PROMPT = (
    "You are a precise document assistant. "
    "Answer ONLY using the provided context. "
    "If the context does not contain enough information, say so clearly. "
    "Do not hallucinate or add information not present in the context."
)

def generate_answer(client, context_chunks: list, query: str, model: str = "llama-3.3-70b-versatile") -> str:
    # build numbered context so the model can reference [1], [2] etc
    numbered_context = ""
    for i, (text, source, page) in enumerate(context_chunks, start=1):
        page_display = page + 1 if isinstance(page, int) else "?"
        numbered_context += f"[{i}] (source: {source}, page {page_display})\n{text}\n\n"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": (
                f"Context chunks:\n{numbered_context}\n"
                f"Question:\n{query}\n\n"
                "Cite the chunk numbers inline in your answer using [1], [2] etc."
            )}
        ]
    )
    return response.choices[0].message.content