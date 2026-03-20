import time

queries = [
    "What problems do customers report about kitchen appliances?",
    "Are there complaints about delivery delays?",
    "What do customers say about DreamRest mattress?",
    "Do customers report product noise issues?",
    "Are there safety concerns mentioned in reviews?"
]


def evaluate_custom_rag(custom_rag):
    print("\n---- Custom FAISS RAG ----")

    for q in queries:
        start = time.time()

        answer = custom_rag.ask(q)

        latency = time.time() - start

        print("\nQuestion:", q)
        print("Answer:", answer)
        print("Latency:", round(latency, 2), "seconds")


def evaluate_langchain_rag(chain):
    print("\n---- LangChain RAG ----")

    for q in queries:
        start = time.time()

        result = chain({"question": q})
        answer = result["answer"]

        latency = time.time() - start

        print("\nQuestion:", q)
        print("Answer:", answer)
        print("Latency:", round(latency, 2), "seconds")