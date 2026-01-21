import streamlit as st
import time
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate

from embeddings.embedder import get_embeddings
from vectorstore.faiss_store import load_faiss_store
from llm.groq_llm import load_llm
from retrieval.retriever import build_retrieval_chain

load_dotenv()

st.title("RAG Document Q&A With Groq And LLaMA-3")

# ---------------- PROMPT ---------------- #

prompt = ChatPromptTemplate.from_template(
"""
You are a research-focused document assistant.

STRICT RULES (must follow all):
1. Use ONLY the information present in the provided context.
2. Do NOT use any external or prior knowledge.
3. Do NOT add assumptions or general explanations.
4. If the answer is NOT explicitly present, respond exactly with:
   "I could not find this information in the provided documents."

ANSWER FORMATTING RULES (VERY IMPORTANT):
- Organise the answer clearly and logically.
- Use short paragraphs, bullet points, or numbered steps ONLY if the document itself implies structure.
- Do NOT add headings unless they reflect the document content.
- Do NOT invent structure that does not exist in the context.

ANSWERING LOGIC:
- If the question asks for a **definition / meaning / term**:
  → Reproduce the definition exactly or with minimal grammatical cleanup.
  → Do NOT summarize or rephrase unnecessarily.

- If the question asks for a **direct fact**:
  → Use the exact sentence or phrase from the context.

- If the question asks for an **explanation, process, or concept**:
  → Combine ONLY the relevant sentences from the context.
  → Present them in a clear, logically ordered manner.
  → You MAY condense repetitive wording, but the meaning must remain unchanged.
  → Do NOT add examples, interpretations, or external clarification.

- If the explanation spans multiple steps or components:
  → Present them as numbered points ONLY if such separation is evident in the text.

<context>
{context}
</context>

Question:
{input}

Answer:
"""
)

# ---------------- RESOURCE LOADING ---------------- #

@st.cache_resource
def load_resources():
    embeddings = get_embeddings()
    vectorstore = load_faiss_store(embeddings)
    llm = load_llm()
    return vectorstore, llm

vectorstore, llm = load_resources()

# ---------------- UI ---------------- #

user_prompt = st.text_input("Enter your query from the research paper")

if user_prompt:
    retrieval_chain = build_retrieval_chain(llm, prompt, vectorstore)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})

    st.write(response["answer"])
