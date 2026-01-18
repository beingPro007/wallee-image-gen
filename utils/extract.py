import json
import re
from typing import Union

def extract_json(text: Union[str, dict]) -> dict:
    """
    Extract JSON from LLM output.
    Supports raw JSON strings, markdown-wrapped JSON, or already-parsed dicts.
    """

    # ✅ If the model already returned parsed JSON, just return it
    if isinstance(text, dict):
        return text

    # ❌ Explicit failure instead of silent crash
    if text is None:
        raise ValueError("extract_json received None (LLM returned no content)")

    text = text.strip()

    # Remove Markdown code fences
    text = re.sub(r"^```(?:json)?\s*|```$", "", text, flags=re.MULTILINE)

    # Fix broken line breaks inside strings
    text = re.sub(r'(?<!\\)"\s*\n\s*', '" ', text)

    # Remove stray carriage returns
    text = text.replace('\r', ' ')

    return json.loads(text)
