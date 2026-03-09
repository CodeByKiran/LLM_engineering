import os
from dotenv import load_dotenv
#import openai
import google.genai as genai
import anthropic
from mistralai import Mistral

import time
load_dotenv()

def call_llm_with_retry(system_prompt, user_message, provider="genai",
                        retries=5, base_delay=5, **kwargs):

    for attempt in range(retries):
        try:
            return call_llm(system_prompt, user_message, provider=provider, **kwargs)

        except Exception as e:
            wait = base_delay * (2 ** attempt)
            print(f"Rate limit hit. Waiting {wait}s...")
            time.sleep(wait)

    raise Exception("Max retries exceeded")


def call_llm(system_prompt, user_message, provider, model=None, temperature=0.2, max_tokens=512):

    
    if provider == "genai":
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        model = model or "gemini-1.5-flash"

        model_client = genai.GenerativeModel(model)

        response = model_client.generate_content(
            f"{system_prompt}\n\n{user_message}"
        )

        return response.text.strip()


    elif provider == "mistral":
        model = model or "mistral-large-latest"

        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

        response = client.chat.complete(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        return response.choices[0].message.content.strip()


    elif provider == "claude":
        model = model or "claude-opus-4-6"

        client = anthropic.Anthropic(
            api_key=os.getenv("CLAUDE_API_KEY")
        )

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        return response.content[0].text.strip()


    else:
        raise ValueError(f"Unknown provider: {provider}")