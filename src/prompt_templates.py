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

