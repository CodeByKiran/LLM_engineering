# Import embedding and chat model interfaces for Mistral
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI

# Import FAISS vector database integration for LangChain
from langchain_community.vectorstores import FAISS

# Conversational RAG chain that combines retrieval + conversation context
from langchain_classic.chains import ConversationalRetrievalChain

# Memory module that stores recent conversation history
from langchain_classic.memory import ConversationBufferWindowMemory

# Used to load API keys from .env file
from dotenv import load_dotenv


# Load environment variables (like MISTRAL_API_KEY)
load_dotenv()


# -----------------------------------------------------------
# FUNCTION: load_vectorstore()
# Purpose:
# Loads the saved FAISS vector database containing review embeddings.
# This database allows semantic search over product reviews.
# -----------------------------------------------------------
def load_vectorstore():

    # Initialize the Mistral embedding model
    # This converts text into numerical vectors (embeddings)
    embeddings = MistralAIEmbeddings(
        model="mistral-embed"
    )

    # Load the FAISS vector index stored locally
    # This index contains embeddings of all review documents
    vectorstore = FAISS.load_local(
        "data/processed/faiss/langchain_faiss",   # location of saved vector database
        embeddings,                               # embedding model used to encode queries
        allow_dangerous_deserialization=True      # required for loading local FAISS objects
    )

    # Return the vectorstore so it can be used by the RAG pipeline
    return vectorstore


# -----------------------------------------------------------
# FUNCTION: build_chat_chain()
# Purpose:
# Creates a conversational RAG pipeline that:
# 1. Receives a user question
# 2. Retrieves relevant reviews from FAISS
# 3. Uses the LLM to generate an answer
# 4. Maintains conversation memory
# -----------------------------------------------------------
def build_chat_chain():

    # Load the vector database containing review embeddings
    vectorstore = load_vectorstore()

    # Initialize the Mistral Large language model
    # This model generates answers based on retrieved context
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0        # temperature=0 makes responses deterministic and factual
    )

    # Create a conversation memory buffer
    # This stores the last k interactions between user and chatbot
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",   # variable name used internally by the chain
        return_messages=True,        # return structured message objects instead of plain text
        k=5                          # remember the last 5 conversation turns
    )

    # Build the Conversational Retrieval Chain
    # This chain combines:
    # - Retrieval from vector database
    # - Conversation history
    # - LLM reasoning
    chain = ConversationalRetrievalChain.from_llm(

        # Language model used for generating responses
        llm=llm,

        # Retriever used to find relevant documents from FAISS
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}   # retrieve top 5 most relevant reviews
        ),

        # Conversation memory module
        memory=memory,

        # Return the retrieved documents along with the answer
        # This is useful for showing sources in the UI
        return_source_documents=True,

        # Disable verbose debugging logs
        verbose=False
    )

    # Return the complete conversational RAG chain
    return chain