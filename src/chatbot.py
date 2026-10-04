"""
RAG Chatbot — interactive CLI
Run: python src/chatbot.py --file data/your_doc.pdf
"""

import argparse
import os
import sys
from rag_pipeline import RAGPipeline


def print_banner():
    print("""
╔══════════════════════════════════════════╗
║           RAG Chatbot  🔍                ║
║  Ask questions about any PDF or .txt     ║
║  Type 'sources' to see retrieved chunks  ║
║  Type 'quit' to exit                     ║
╚══════════════════════════════════════════╝
""")


def chat_loop(rag: RAGPipeline):
    print("\nReady! Ask a question about your document.\n")
    show_sources = True

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if question.lower() == "sources":
            show_sources = not show_sources
            print(f"Source display: {'ON' if show_sources else 'OFF'}\n")
            continue

        if question.lower().startswith("search "):
            # Debug: show raw retrieval scores
            query = question[7:]
            results = rag.search(query)
            print("\n── Similarity search results ──")
            for doc, score in results:
                print(f"  Score {score:.3f} | {doc.page_content[:100]}...")
            print()
            continue

        print("\nAssistant: ", end="", flush=True)
        answer = rag.ask(question, show_sources=show_sources)
        print(answer)
        print()


def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot over your documents")
    parser.add_argument("--file", type=str, help="Path to PDF or .txt file")
    parser.add_argument("--index", type=str, default="", help="Path to saved FAISS index")
    parser.add_argument("--save-index", type=str, default="", help="Save FAISS index after building")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
         print("ERROR: Set your GROQ_API_KEY environment variable first.")
         print("  export GROQ_API_KEY='gsk_...'")
         sys.exit(1)

    print_banner()

    rag = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        model=args.model,
    )

    if args.index:
        rag.load_index(args.index)
    elif args.file:
        rag.load_document(args.file)
        rag.build_index(save_path=args.save_index or None)
    else:
        print("ERROR: Provide --file or --index\n")
        parser.print_help()
        sys.exit(1)

    chat_loop(rag)


if __name__ == "__main__":
    main()
