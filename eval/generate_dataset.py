import os
import json
import random
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# ========= CONFIG =========
PDF_FILES = [
    "/Users/hyashwanth/Desktop/GEN AI Projects/RAG_Projects/RAG_Project_Org_level_Phase1/data/finance/bosch-annual-report-2024.pdf",
    "/Users/hyashwanth/Desktop/GEN AI Projects/RAG_Projects/RAG_Project_Org_level_Phase1/data/marketing/bosch-in-india-golden-book.pdf"
]
NUM_SAMPLES = 30
OUTPUT_FILE = "benchmark_data.json"
# ==========================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def generate_qa_from_chunk(chunk):

    prompt = f"""
Generate ONE factual question and its exact answer from the context.

Context:
{chunk}

Return ONLY valid JSON with this format:
{{
  "question": "...",
  "answer": "..."
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"}   # 🔥 forces JSON
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)

    except json.JSONDecodeError:
        print("⚠ JSON parsing failed. Raw response:")
        print(content)
        raise

def main():
    print("Extracting PDF text...")

    all_chunks = []

    for path in PDF_FILES:
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    sample_chunks = random.sample(all_chunks, NUM_SAMPLES)

    dataset = []

    print("Generating Q&A pairs...")

    for chunk in sample_chunks:
        qa = generate_qa_from_chunk(chunk)

        dataset.append({
            "question": qa["question"],
            "ground_truth": qa["answer"],
            "contexts": [chunk]
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=4)

    print(f"Benchmark dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()