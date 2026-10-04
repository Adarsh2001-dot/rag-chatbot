"""
Evaluate different chunking strategies.
Great to run and screenshot for your GitHub README.

Usage: python src/evaluate.py --file data/sample.txt
"""

import argparse
import os
import time
from rag_pipeline import RAGPipeline

TEST_QUESTIONS = [
    "What is the main topic of this document?",
    "What are the key findings or conclusions?",
    "What methods or techniques are described?",
]

STRATEGIES = [
    {"chunk_size": 200, "chunk_overlap": 20,  "label": "Small chunks  (200 chars)"},
    {"chunk_size": 500, "chunk_overlap": 50,  "label": "Medium chunks (500 chars) ← recommended"},
    {"chunk_size": 1000,"chunk_overlap": 100, "label": "Large chunks (1000 chars)"},
]


def evaluate(file_path: str):
    print(f"\n{'='*60}")
    print("RAG Chunking Strategy Evaluation")
    print(f"File: {file_path}")
    print(f"{'='*60}\n")

    results = []

    for strategy in STRATEGIES:
        print(f"\n── {strategy['label']} ──────────────────────────")

        rag = RAGPipeline(
            chunk_size=strategy["chunk_size"],
            chunk_overlap=strategy["chunk_overlap"],
        )
        n = rag.load_document(file_path)
        rag.build_index()

        for q in TEST_QUESTIONS:
            start = time.time()
            answer = rag.ask(q, show_sources=False)
            elapsed = time.time() - start

            print(f"\nQ: {q}")
            print(f"A: {answer[:200]}...")
            print(f"   ↳ {n} chunks | {elapsed:.2f}s")

            results.append({
                "strategy": strategy["label"],
                "question": q,
                "answer_len": len(answer),
                "time": elapsed,
                "n_chunks": n,
            })

    # Summary table
    print(f"\n\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Strategy':<35} {'Chunks':>7} {'Avg time':>10}")
    print("-" * 55)

    seen = set()
    for r in results:
        if r["strategy"] not in seen:
            seen.add(r["strategy"])
            times = [x["time"] for x in results if x["strategy"] == r["strategy"]]
            avg_t = sum(times) / len(times)
            print(f"{r['strategy']:<35} {r['n_chunks']:>7} {avg_t:>9.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to PDF or .txt file")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first.")
        exit(1)

    evaluate(args.file)
