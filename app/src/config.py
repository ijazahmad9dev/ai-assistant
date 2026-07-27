import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH="/home/ahmad/Documents/Projects/NextBridge internship/Project02/ai-assistant/data"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MILVUS_URI = "app/vectordb/milvus.db"
MILVUS_COLLECTION = "rag_docs"


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

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