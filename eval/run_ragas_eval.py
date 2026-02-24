import os
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from rag.chat import generate_answer

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# -----------------------------
# Fix embed_query for RAGAS
# -----------------------------
class RagasEmbeddingWrapper(OpenAIEmbeddings):
    def embed_query(self, text: str):
        return self.embed_documents([text])[0]


def main():

    print("Loading benchmark dataset...")

    with open("eval/benchmark_data.json", "r") as f:
        benchmark_data = json.load(f)

    questions, answers, contexts, ground_truths = [], [], [], []

    print("Generating answers using production RAG...")

    for idx, item in enumerate(benchmark_data):
        print(f"→ {idx+1}/{len(benchmark_data)}")

        question = item["question"]
        ground_truth = item["ground_truth"]

        response = generate_answer(question, history=[])

        answer = response.get("answer")

        if answer is None:
            raise ValueError("generate_answer() must return 'answer'")

        retrieved_contexts = response.get("contexts", [])

        if not isinstance(retrieved_contexts, list):
            raise ValueError("'contexts' must be a list of strings")

        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(ground_truth)

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    print("Running RAGAS evaluation...")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    embeddings = RagasEmbeddingWrapper(
        model="text-embedding-3-small"
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()

    os.makedirs("eval/results", exist_ok=True)

    df.to_json(
        "eval/results/ragas_results.json",
        orient="records",
        indent=4,
    )

    print("\nEvaluation complete!\n")
    print("Mean Scores:")
    print(df.mean(numeric_only=True))


if __name__ == "__main__":
    main()