import json
import re

def extract_json(text: str):
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for block in matches:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found")
