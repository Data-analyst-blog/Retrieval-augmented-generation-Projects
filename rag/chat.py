from openai import OpenAI
from rag.retriever import retrieve
from rag.reranker import rerank
from rag.utils import build_citations, ask_llm_confidence
from rag.prompt import build_prompt
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()   

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def calculate_hybrid_confidence(vector_scores, rerank_scores, k=5):

    if not vector_scores or not rerank_scores:
        return 0.0

    # Convert L2 distance → similarity
    vector_similarities = [1 / (1 + s) for s in vector_scores]

    avg_vector = np.mean(vector_similarities)
    avg_rerank = np.mean(rerank_scores)

    # Hybrid weighting (reranker more important)
    confidence = (
        0.4 * avg_vector +
        0.6 * avg_rerank
    )

    return round(float(min(max(confidence, 0), 1)), 3)

def generate_answer(query, history):

    docs = retrieve(query, k=20)

    if not docs:
        return {
            "answer": "No relevant documents found.",
            "contexts": [],
            "citations": [],
            "confidence": 0.0
        }

    vector_scores = [doc["similarity_score"] for doc in docs]

    docs = rerank(query, docs, top_k=5)

    rerank_scores = [doc["rerank_score"] for doc in docs]

    confidence = calculate_hybrid_confidence(
        vector_scores=vector_scores[:5],
        rerank_scores=rerank_scores,
        k=5
    )

    prompt = build_prompt(query, docs, history)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    llm_confidence = ask_llm_confidence(answer, docs, client)

    confidence = round(
        0.7 * confidence + 0.3 * llm_confidence,
        3
    )

    citations = build_citations(docs)

    # -----------------------------
    # SAFE CONTEXT EXTRACTION
    # -----------------------------
    def extract_text(doc):
        if isinstance(doc, dict):
            return (
                doc.get("content")
                or doc.get("text")
                or doc.get("page_content")
                or ""
            )
        return ""

    contexts = [extract_text(doc) for doc in docs]

    return {
        "answer": answer,
        "contexts": contexts,   # 🔥 Required for RAGAS
        "citations": citations,
        "confidence": confidence
    }