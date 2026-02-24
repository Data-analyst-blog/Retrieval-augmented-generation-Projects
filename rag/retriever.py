import faiss
import json
import numpy as np
from ingestion.embedder import model
import os

def load_index():
    index = faiss.read_index("vectorstore/faiss_index/index.faiss")

    with open("vectorstore/metadata.json", "r") as f:
        metadata = json.load(f)

    return index, metadata

index, metadata = load_index()
last_loaded = 0

def check_reload():
    global index, metadata, last_loaded

    current_modified = os.path.getmtime("vectorstore/faiss_index/index.faiss")

    if current_modified != last_loaded:
        index, metadata = load_index()
        last_loaded = current_modified
        print("🔄 Index reloaded")

def retrieve(query, k=5):
    check_reload()

    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    results = []

    for i, idx in enumerate(indices[0]):
        doc_meta = metadata[idx]

        results.append({
            "text": doc_meta["text"],
            "file_name": doc_meta.get("file_name"),
            "department": doc_meta.get("department"),
            "file_type": doc_meta.get("file_type"),
            "source_type": doc_meta.get("source_type"),
            "ingestion_time": doc_meta.get("ingestion_time"),
            "similarity_score": float(distances[0][i])
        })

    return results