import pandas as pd
import time
from tqdm import tqdm

import sys
sys.path.append("..")
from src.llm_client import call_llm
from src.prompt_templates import (
    SENTIMENT_SYSTEM,
    build_sentiment_prompt,
    ENTITY_SYSTEM,
    build_entity_prompt,
    SUMMARIZATION_SYSTEM,
    build_summarization_prompt,
    TOPIC_LABEL_SYSTEM,
    build_topic_label_prompt
)
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

    from src.prompt_templates import (
        SUMMARIZATION_SYSTEM,
        build_summarization_prompt
    )

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        user_msg = build_summarization_prompt(row["review_text"])

        summary = call_llm(
            SUMMARIZATION_SYSTEM,
            user_msg,
            provider=provider
        )

        results.append({
            "review_id": row["review_id"],
            "llm_summary": summary.strip()
        })

        time.sleep(delay)

    return pd.DataFrame(results)





############################################
#  ENTITY EXTRACTION
############################################

def batch_extract_entities(df, provider="mistral", delay=0.5):

    from src.prompt_templates import (
        ENTITY_SYSTEM,
        build_entity_prompt
    )

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        user_msg = build_entity_prompt(row["review_text"])

        raw = call_llm(
            ENTITY_SYSTEM,
            user_msg,
            provider=provider
        )

        parsed = safe_parse_json(raw)

        parsed["review_id"] = row["review_id"]

        results.append(parsed)

        time.sleep(delay)

    return pd.DataFrame(results)





############################################
#  TOPIC LABELING
############################################
def label_topic(keywords, provider="mistral"):

    from src.prompt_templates import (
        TOPIC_LABEL_SYSTEM,
        build_topic_label_prompt
    )

    user_msg = build_topic_label_prompt(keywords)

    label = call_llm(
        TOPIC_LABEL_SYSTEM,
        user_msg,
        provider=provider
    )

    return label.strip()