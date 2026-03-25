from pathlib import Path

BASE_DIR = Path(__file__).parent
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
PDF_PATH = BASE_DIR / "python_guide.pdf"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "deepseek-chat"
LLM_BASE_URL = "https://api.deepseek.com/v1"
LLM_TEMPERATURE = 0.7

SEARCH_K = 3