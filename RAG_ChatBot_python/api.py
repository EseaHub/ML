import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from config import LLM_MODEL, LLM_BASE_URL, LLM_TEMPERATURE

load_dotenv()

def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("❌ API ключ не найден! Проверьте файл .env")
    
    return ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=api_key,
        openai_api_base=LLM_BASE_URL,
        temperature=LLM_TEMPERATURE
    )