def build_citations(docs):

    citation_map = {}

    for doc in docs:
        file_name = doc.get("file_name")

        if file_name not in citation_map:
            citation_map[file_name] = {
                "title": file_name,
                "url": file_name if doc.get("source_type") == "web" else f"/docs/{file_name}"
            }

    citations = []
    for i, (key, value) in enumerate(citation_map.items(), start=1):
        value["id"] = i
        citations.append(value)

    return citations

def ask_llm_confidence(answer, docs, client):

    context_text = "\n\n".join([d["text"] for d in docs])

    evaluation_prompt = f"""
You are evaluating answer grounding.

Context:
{context_text}

Answer:
{answer}

On a scale of 0 to 1, how confident are you that the answer is fully supported by the context?
Return ONLY a number between 0 and 1.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": evaluation_prompt}]
    )

    try:
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0), 1)
    except:
        return 0.5