import streamlit as st
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import os

load_dotenv()

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="HomeNest Review Intelligence",
    page_icon="🏠",
    layout="wide"
)

# ── Session State Initialisation ─────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "sources" not in st.session_state:
    st.session_state.sources = []

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:

    st.title("⚙️ Settings")

    model_choice = st.selectbox(
        "Mistral Model",
        [
            "mistral-large-latest",
            "mistral-small-latest"
        ]
    )

    st.divider()

    st.subheader("📋 Sample Questions")

    sample_qs = [
        "What are the most common product defects?",
        "Are there any safety concerns I should know about?",
        "Which product has the best customer reviews?",
        "What do customers say about delivery and packaging?"
    ]

    for q in sample_qs:
        if st.button(q, use_container_width=True):
            st.session_state.prefill_question = q

    st.divider()

    if st.button("🗑 Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.chain = None
        st.session_state.sources = []
        st.rerun()

    st.divider()

    st.subheader("📚 Retrieved Sources")

    for src in st.session_state.sources:
        with st.expander(f"Review #{src['id']} — {src['product']}"):
            st.write(f"⭐ {src['stars']} stars")
            st.write(src["text"][:300] + "...")

# ── Main UI ──────────────────────────────────────────────
st.title("🏠 HomeNest Review Intelligence Chatbot")
st.caption("Powered by RAG using Mistral AI")

# Show chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Load Vector Store ────────────────────────────────────
@st.cache_resource
def get_vectorstore():

    embeddings = MistralAIEmbeddings(
        model="mistral-embed"
    )

    return LangFAISS.load_local(
        "../data/processed/langchain_faiss/",
        embeddings,
        allow_dangerous_deserialization=True
    )

# ── Build RAG Chain ──────────────────────────────────────
def build_chain(model_name):

    vectorstore = get_vectorstore()

    # LLM
    llm = ChatMistralAI(
        model=model_name,
        temperature=0
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    # Memory
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
        output_key="answer"
    )

    # RAG Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        chain_type="stuff"
    )

    return chain

# ── Handle User Input ────────────────────────────────────
prefill = st.session_state.pop("prefill_question", "")

user_input = st.chat_input("Ask about HomeNest customer reviews...") or prefill

if user_input:

    if st.session_state.chain is None:
        st.session_state.chain = build_chain(model_choice)

    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Searching reviews and generating answer..."):

            result = st.session_state.chain.invoke(
                {"question": user_input}
            )

            answer = result["answer"]
            sources = result.get("source_documents", [])

            st.write(answer)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

            st.session_state.sources = [
            {
              "id": d.metadata.get("review_id", "N/A"),
              "product": d.metadata.get("product_name", "Unknown"),
              "stars": d.metadata.get("star_rating", "N/A"),
              "text": d.page_content
           }
    for d in sources
]

    st.rerun()