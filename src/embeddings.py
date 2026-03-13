import os
import numpy as np
import faiss
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