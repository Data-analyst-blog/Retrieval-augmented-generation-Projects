def build_prompt(query, retrieved_docs, history):

    context = "\n\n".join(
        [f"Source: {doc['file_name']}\n{doc['text']}" for doc in retrieved_docs]
    )

    prompt = f"""
You are an internal company assistant and you will answer questions based on provided context only.

Chat History:
{history}

Context:
{context}

Question:
{query}

You must answer using only the context provided. 
Do NOT mention document names or sources in the answer.
"""

    return prompt