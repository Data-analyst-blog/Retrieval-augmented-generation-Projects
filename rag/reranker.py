from sentence_transformers import CrossEncoder
import numpy as np
from scipy.special import expit

reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, results, top_k=5):

    if not results:
        return results

    pairs = [(query, r["text"]) for r in results]

    raw_scores = reranker_model.predict(pairs)
    normalized_scores = expit(raw_scores)

    for i in range(len(results)):
        results[i]["rerank_score"] = float(normalized_scores[i])

    results.sort(key=lambda x: x["rerank_score"], reverse=True)

    return results[:top_k]