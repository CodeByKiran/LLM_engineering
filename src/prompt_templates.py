# src/prompt_templates.py

#####################################################################
##########               SENTIMENT PROMPT                  ##########
#####################################################################
SENTIMENT_SYSTEM = """
You are an expert product review sentiment analyst.

Return ONLY valid JSON.
"""


def build_sentiment_prompt(review_text):

    prompt = f"""
Analyze the following product review.

Return JSON in this format:

{{
 "llm_sentiment": "Positive | Neutral | Negative",
 "llm_confidence": float,
 "llm_key_issue": "short phrase or null",
 "llm_is_safety": true | false
}}

Review:
"{review_text}"
"""

    return prompt



#####################################################################
##########               ENTITY PROMPT                     ##########
#####################################################################
ENTITY_SYSTEM = """
You are a named entity extractor for product reviews.
Extract entities and return ONLY valid JSON — no explanation.
"""

def build_entity_prompt(review_text):
    return f"""
Extract the following from this review:

- product_features: list of product aspects mentioned (e.g. motor, handle, delivery)
- problems: list of problems mentioned (empty list if none)
- praise: list of things praised (empty list if none)
- mentioned_timeframe: how long before the issue occurred (e.g. '2 weeks', null if not mentioned)

Review: "{review_text}"

Respond ONLY with JSON.

Example:
{{
"product_features": ["motor", "blade"],
"problems": ["motor stopped", "blade cracked"],
"praise": [],
"mentioned_timeframe": "2 weeks"
}}
"""



#####################################################################
##########               Topic Labeling                    ##########
#####################################################################
TOPIC_LABEL_SYSTEM = """
You are a topic labeler for customer review analytics.

Given a list of keywords from a topic cluster, generate a short,
clear label (2–4 words) that captures what customers are discussing.

Respond ONLY with the label.
"""

def build_topic_label_prompt(keywords):

    return f"""
Keywords: {keywords}

Label:
"""



#####################################################################
##########                 RAG                             ##########
#####################################################################

RAG_SYSTEM = """
You are HomeNest's customer review intelligence assistant.

You answer questions about product quality, customer sentiment,
and common issues based ONLY on the customer reviews provided.

Rules:
- Only use information present in the provided reviews
- If the context does not contain enough information, say so clearly
- Always cite which review IDs support your answer
- Be concise: answer in 3–5 sentences maximum
- Do NOT make up product specifications or policies
"""

RAG_USER = """
Context (relevant customer reviews):
---
{context_block}
---

Question:
{user_question}
"""