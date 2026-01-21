from langchain_community.vectorstores import FAISS
import os

VECTOR_PATH = "vectorstore/index"

def create_faiss_store(documents, embeddings):
    vectorstore = FAISS.from_documents(documents, embeddings)
    os.makedirs(VECTOR_PATH, exist_ok=True)
    vectorstore.save_local(VECTOR_PATH)

def load_faiss_store(embeddings):
    return FAISS.load_local(
        VECTOR_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
