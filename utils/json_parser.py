import json
import re

def safe_parse_json(llm_output):
    """Extract JSON even if the model returns extra text."""

    try:
        return json.loads(llm_output)

    except json.JSONDecodeError:

        match = re.search(r'\{.*?\}', llm_output, re.DOTALL)

        if match:
            return json.loads(match.group())

        return {
            "error": "unparseable",
            "raw": llm_output
        }