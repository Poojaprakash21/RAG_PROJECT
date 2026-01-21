import os
from collections import defaultdict

from ingestion.loader import load_documents
from ingestion.chunker import split_documents
from embeddings.embedder import get_embeddings
from vectorstore.faiss_store import create_faiss_store
from utils.json_writer import write_json

PDF_DIR = "research_papers"
OUTPUT_DIR = "outputs"

LOADER_DIR = f"{OUTPUT_DIR}/loader"
CHUNK_DIR = f"{OUTPUT_DIR}/chunking"
EMBED_DIR = f"{OUTPUT_DIR}/embeddings"


def run_pipeline():
    os.makedirs(LOADER_DIR, exist_ok=True)
    os.makedirs(CHUNK_DIR, exist_ok=True)
    os.makedirs(EMBED_DIR, exist_ok=True)

    # ======================================================
    # 01 — LOAD DOCUMENTS
    # ======================================================
    print("🔹 Loading PDFs...")
    documents = load_documents(PDF_DIR)

    loader_group = defaultdict(list)

    for d in documents:
        pdf_name = os.path.basename(d.metadata.get("source"))
        loader_group[pdf_name].append({
            "page": d.metadata.get("page"),
            "text_preview": d.page_content[:500]
        })

    for pdf, pages in loader_group.items():
        write_json(
            {
                "pdf_name": pdf,
                "total_pages": len(pages),
                "pages": pages
            },
            f"{LOADER_DIR}/{pdf.replace('.pdf', '.json')}"
        )

    # ======================================================
    # 02 — CHUNK DOCUMENTS (GLOBAL CHUNK IDs)
    # ======================================================
    print("🔹 Chunking documents...")
    chunks = split_documents(documents)

    chunk_group = defaultdict(list)

    for chunk_id, c in enumerate(chunks):
        pdf_name = os.path.basename(c.metadata.get("source"))
        chunk_group[pdf_name].append({
            "chunk_id": chunk_id,
            "page": c.metadata.get("page"),
            "chunk_size": len(c.page_content),
            "chunk_text": c.page_content
        })

    for pdf, pdf_chunks in chunk_group.items():
        write_json(
            {
                "pdf_name": pdf,
                "total_chunks": len(pdf_chunks),
                "chunks": pdf_chunks
            },
            f"{CHUNK_DIR}/{pdf.replace('.pdf', '.json')}"
        )

    # ======================================================
    # 03 — EMBEDDINGS (TEXT + VECTOR PER CHUNK)
    # ======================================================
    print("🔹 Creating embeddings...")
    embeddings = get_embeddings()

    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)

    embed_group = defaultdict(list)

    for chunk_id, (c, vector) in enumerate(zip(chunks, vectors)):
        pdf_name = os.path.basename(c.metadata.get("source"))
        embed_group[pdf_name].append({
            "chunk_id": chunk_id,
            "chunk_text": c.page_content,
            "vector": vector
        })

    for pdf, data in embed_group.items():
        write_json(
            {
                "pdf_name": pdf,
                "embedding_model": embeddings.model_name,
                "dimension": len(data[0]["vector"]),
                "chunks": data
            },
            f"{EMBED_DIR}/{pdf.replace('.pdf', '.json')}"
        )

    # ======================================================
    # 04 — FAISS VECTOR STORE (GLOBAL)
    # ======================================================
    print("🔹 Building FAISS vector store...")
    create_faiss_store(chunks, embeddings)

    print("✅ Pipeline completed with full, accurate observability.")


if __name__ == "__main__":
    run_pipeline()
