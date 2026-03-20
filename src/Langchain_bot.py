from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS as LangFAISS


def build_chat_chain():


    # Load embedding model (Mistral)
    embeddings_model = MistralAIEmbeddings(
        model="mistral-embed"
    )


    # Load FAISS vector database
    vectorstore = LangFAISS.load_local(
        "../data/processed/langchain_faiss/",
        embeddings_model,
        allow_dangerous_deserialization=True
    )


    # Load Mistral LLM
    llm = ChatMistralAI(
        model="mistral-small",
        temperature=0
    )


    # Conversation memory (last 5 interactions)
    memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
    )


    # Build Conversational RAG chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )


    return chain