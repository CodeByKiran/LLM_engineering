# src/prompt_templates.py
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
