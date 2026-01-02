# llm.py
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv(override=True)

CONTROL_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.0

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model = CONTROL_MODEL,
            temperature = TEMPERATURE
        )
    return _llm