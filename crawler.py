import os
import json
import shutil
import numpy as np
import faiss
from datetime import datetime

from ingestion.loader import load_files
from ingestion.parser import parse_file
from ingestion.web_crawler import crawl_website
from ingestion.chunker import chunk_text
from ingestion.embedder import generate_embeddings

VECTOR_PATH = "vectorstore/faiss_index"
META_PATH = "vectorstore/metadata.json"
LOCK_FILE = "crawler.lock"

def run_ingestion():

    print("🚀 Starting ingestion...")

    documents = load_files("data")

    all_chunks = []
    metadata = []
    chunk_id = 0

    for doc in documents:

        file_path = doc["file_path"]
        file_name = doc["file_name"]
        department = doc["department"]

        file_extension = file_name.split(".")[-1].lower()

        print(f"Processing: {file_name}")

        # 🔹 WEBSITE LINKS (special handling)
        if file_name == "website_links.txt":
            with open(file_path, "r") as f:
                urls = f.readlines()

            for url in urls:
                url = url.strip()
                if not url:
                    continue

                text = crawl_website(url)
                chunks = chunk_text(text)

                for chunk in chunks:
                    all_chunks.append(chunk)

                    metadata.append({
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "department": department,
                        "file_name": url,
                        "file_type": "website",
                        "source_type": "web",
                        "ingestion_time": str(datetime.utcnow())
                    })

                    chunk_id += 1

        else:
            # 🔹 FILE PARSING
            text = parse_file(file_path)

            if not text:
                continue

            chunks = chunk_text(text)

            for chunk in chunks:
                all_chunks.append(chunk)

                metadata.append({
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "department": department,
                    "file_name": file_name,
                    "file_type": file_extension,
                    "source_type": "file",
                    "ingestion_time": str(datetime.utcnow())
                })

                chunk_id += 1

    print(f"📄 Total chunks created: {len(all_chunks)}")

    if not all_chunks:
        print("No content found.")
        return

    # 🔹 EMBEDDING
    embeddings = generate_embeddings(all_chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # 🔹 FULL REINDEX (delete old)
    if os.path.exists(VECTOR_PATH):
        shutil.rmtree(VECTOR_PATH)

    os.makedirs(VECTOR_PATH)

    faiss.write_index(index, f"{VECTOR_PATH}/index.faiss")

    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    print("✅ Ingestion completed successfully.")


if __name__ == "__main__":

    if os.path.exists(LOCK_FILE):
        print("Crawler already running.")
        exit()

    open(LOCK_FILE, "w").close()

    try:
        run_ingestion()
    finally:
        os.remove(LOCK_FILE)