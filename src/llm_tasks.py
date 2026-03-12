import pandas as pd
import time
from tqdm import tqdm

import sys
sys.path.append("..")
from src.llm_client import call_llm
from src.prompt_templates import SENTIMENT_SYSTEM,build_sentiment_prompt
from utils.json_parser import safe_parse_json


############################################
# SENTIMENT ANALYSIS
############################################

def batch_sentiment(df, provider="mistral", delay=0.5):

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        #user_msg = SENTIMENT_USER.format(review_text = row["review_text"])
        user_msg = build_sentiment_prompt(row["review_text"])
        raw = call_llm(SENTIMENT_SYSTEM,user_msg,provider=provider)
        parsed = safe_parse_json(raw)
        parsed["review_id"] = row["review_id"]
        results.append(parsed)
        time.sleep(delay) #Rate Limiting 

    return pd.DataFrame(results)


############################################
#  SUMMARIZATION
############################################

def batch_summarize(df, provider="mistral", delay=0.5):

    SYSTEM = """
You are a concise product review summarizer for an e-commerce analytics team.

Rules:
- Write exactly ONE sentence (max 20 words)
- Include the product aspect mentioned
- Include what the customer felt
- Do NOT start with 'The customer' or 'This review'
- Do NOT add information not in the review
"""

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        user = f"""
Summarize this review:

{row['review_text']}
"""

        summary = call_llm(
            SYSTEM,
            user,
            provider=provider
        )

        results.append({
            "review_id": row["review_id"],
            "llm_summary": summary.strip()
        })

        time.sleep(delay)

    return pd.DataFrame(results)


############################################
# 3️⃣ ENTITY EXTRACTION
############################################

def batch_extract_entities(df, provider="mistral", delay=0.5):

    SYSTEM = """
You are a named entity extractor for product reviews.

Extract entities and return ONLY valid JSON.

Fields:
- product_features
- problems
- praise
- mentioned_timeframe
"""

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        user = f"""
Extract the following from this review:

- product_features: list of product aspects mentioned
- problems: list of problems mentioned (empty list if none)
- praise: list of things praised (empty list if none)
- mentioned_timeframe: how long before issue occurred (or null)

Review:
"{row['review_text']}"

Return ONLY JSON.
"""

        raw = call_llm(
            SYSTEM,
            user,
            provider=provider
        )

        parsed = safe_parse_json(raw)

        parsed["review_id"] = row["review_id"]

        results.append(parsed)

        time.sleep(delay)

    return pd.DataFrame(results)


############################################
# 4️⃣ TOPIC LABELING
############################################

def label_topic(keywords, provider="mistral"):

    SYSTEM = """
You are a topic labeler for customer review analytics.

Given keywords from a topic cluster,
generate a short label (2–4 words).

Return ONLY the label.
"""

    user = f"""
Keywords: {keywords}

Label:
"""

    label = call_llm(
        SYSTEM,
        user,
        provider=provider
    )

    return label.strip()