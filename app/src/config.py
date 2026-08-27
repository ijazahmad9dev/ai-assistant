import os
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
DATA_PATH = BASE_DIR / "data"

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "app/data/faiss_index"


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"

# SMTP (Gmail, for now)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD")

GM_EMAIL = os.getenv("GM_EMAIL")  
FOOD_DEPT_EMAIL = os.getenv("FOOD_DEPT_EMAIL")
MIS_EMAIL = os.getenv("MIS_EMAIL")

IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEXTBRIDGE_DOMAIN = "nextbridge.com" 

OLLAMA_BASE_URL = "https://relation-creature-tap-bradley.trycloudflare.com"
OLLAMA_MODEL = "gpt-oss:latest"

EMBED_MODEL_ID = "qwen3-embedding:latest"

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "nextbridge-crag-eval")

# LangGraph/LangChain read these directly from the process environment,
# so mirror them there at startup
os.environ["LANGCHAIN_TRACING_V2"] = LANGSMITH_TRACING
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT