import json
import re

def extract_json(text: str) -> dict:
    """
    Extract JSON from LLM output, remove code fences, fix line breaks in strings.
    """
    text = text.strip()

    # Remove Markdown code fences (```json or ```)
    text = re.sub(r"^```json\s*|```$", "", text, flags=re.MULTILINE)
    text = re.sub(r'(?<!\\)"\s*\n\s*', '" ', text)

    # Remove any stray \r characters
    text = text.replace('\r', ' ')

    # Now parse
    return json.loads(text)
