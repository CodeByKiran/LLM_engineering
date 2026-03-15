import os
import numpy as np
import faiss
import pandas as pd 
from dotenv import load_dotenv
from mistralai import Mistral
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))


# --------------------------------
# Single embedding
# --------------------------------
def get_embedding(text):

    text = text.replace("\n", " ")

    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )

    return np.array(response.data[0].embedding)


# --------------------------------
# Dataframe embeddings
# --------------------------------
def embed_dataframe(df, text_col="review_text", batch_size=50):

    embeddings = []

    for i in range(0, len(df), batch_size):

        batch = df[text_col].iloc[i:i+batch_size].tolist()

        response = client.embeddings.create(
            model="mistral-embed",
            inputs=batch
        )

        embeddings.extend([r.embedding for r in response.data])

    return np.array(embeddings)


#---------------------------
# FAISS 
#----------------------------

def build_faiss_index(embeddings, df, save_dir="../data/processed/faiss/"):

    os.makedirs(save_dir, exist_ok=True)

    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed_embeddings = embeddings / norms

    dim = normed_embeddings.shape[1]

    # Create index
    index = faiss.IndexFlatIP(dim)

    # Add vectors
    index.add(normed_embeddings.astype("float32"))

    # Save index
    faiss.write_index(index, f"{save_dir}/reviews.index")

    # Save metadata
    df.to_csv(f"{save_dir}/metadata.csv", index=False)

    print(f"Index built with {index.ntotal} vectors")

    return index


def faiss_search(query_text, index, df, top_k=5):

    query_embedding = get_embedding(query_text).reshape(1, -1).astype("float32")

    # Normalize query
    query_embedding /= np.linalg.norm(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    results = df.iloc[indices[0]].copy()
    results["score"] = distances[0]

    return results