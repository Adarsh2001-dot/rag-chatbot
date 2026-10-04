"""
RAG Pipeline — core logic
Handles: chunking → embedding → vector store → retrieval → generation
"""

import os
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 3,
        model: str = "openai/gpt-oss-20b",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.model = model

        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.chunks = []
        self.vectorstore = None

    def load_document(self, path: str) -> int:
        print(f"\n[1/3] Loading document: {path}")
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path)
        else:
            raise ValueError("Supported formats: .pdf, .txt")

        pages = loader.load()
        print(f"    Loaded {len(pages)} page(s)")

        print(f"\n[2/3] Chunking (size={self.chunk_size}, overlap={self.chunk_overlap})")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunks = splitter.split_documents(pages)
        avg_len = sum(len(c.page_content) for c in self.chunks) // len(self.chunks)
        print(f"    Created {len(self.chunks)} chunks (avg {avg_len} chars each)")
        return len(self.chunks)

    def build_index(self, save_path: Optional[str] = None):
        if not self.chunks:
            raise RuntimeError("Call load_document() first.")
        print(f"\n[3/3] Embedding {len(self.chunks)} chunks and building FAISS index...")
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
        if save_path:
            self.vectorstore.save_local(save_path)
            print(f"    Index saved to: {save_path}")
        print("    Index ready. Ask away!\n")

    def load_index(self, save_path: str):
        print(f"Loading saved index from: {save_path}")
        self.vectorstore = FAISS.load_local(
            save_path, self.embeddings, allow_dangerous_deserialization=True
        )
        print("Index loaded. Ask away!\n")

    def ask(self, question: str, show_sources: bool = True) -> str:
        if not self.vectorstore:
            raise RuntimeError("Call build_index() or load_index() first.")

        # Retrieve top-k chunks
        results = self.vectorstore.similarity_search_with_score(question, k=self.top_k)
        context = "\n\n".join([doc.page_content for doc, _ in results])

        if show_sources:
            print("\n── Sources retrieved ──────────────────────────────────────")
            for i, (doc, score) in enumerate(results, 1):
                page = doc.metadata.get("page", "?")
                preview = doc.page_content[:120].replace("\n", " ")
                print(f"  [{i}] Score {score:.3f} | Page {page}: {preview}...")
            print("───────────────────────────────────────────────────────────\n")

        # Call Groq directly
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer using ONLY the context provided. If the answer isn't in the context, say: 'I don't have enough information to answer that.'"
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def search(self, query: str) -> list:
        if not self.vectorstore:
            raise RuntimeError("Build index first.")
        return self.vectorstore.similarity_search_with_score(query, k=self.top_k)