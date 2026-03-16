import faiss
import numpy as np
import pandas as pd

from src.embeddings import get_embedding
from src.llm_client import call_llm
from src.prompt_templates import RAG_SYSTEM, RAG_USER


class ReviewRAG:

    def __init__(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)
        self.df = pd.read_csv(metadata_path)

    def retrieve(self, question, top_k=5):

        q_emb = get_embedding(question).reshape(1, -1).astype("float32")

        q_emb /= np.linalg.norm(q_emb)

        scores, idxs = self.index.search(q_emb, top_k)

        results = self.df.iloc[idxs[0]].copy()

        results["score"] = scores[0]

        return results

    def build_context(self, retrieved_df):

        lines = []

        for _, row in retrieved_df.iterrows():

            lines.append(
                f"[Review #{row['review_id']} | {row['product_name']} | {row['star_rating']} stars]\n"
                f"{row['review_text']}"
            )

        return "\n\n".join(lines)

    def answer(self, question, provider="mistral", top_k=5):

        retrieved = self.retrieve(question, top_k=top_k)

        context = self.build_context(retrieved)

        user_msg = RAG_USER.format(
            context_block=context,
            user_question=question
        )

        answer = call_llm(
            RAG_SYSTEM,
            user_msg,
            provider=provider
        )

        return {
            "question": question,
            "answer": answer,
            "source_ids": retrieved["review_id"].tolist(),
            "top_reviews": retrieved[
                ["review_id", "product_name", "review_text", "score"]
            ]
        }