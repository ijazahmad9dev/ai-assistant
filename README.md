# NextBridge Assistant

An internal RAG-powered chatbot for NextBridge employees. Answers company/policy questions from internal documents, falls back to a scoped web search, and handles HR-style workflows (leave, WFH, meal subscription, MIS complaints) end-to-end via email — including reading and surfacing department replies.

## Features

- **Document Q&A (RAG)** — ingests internal PDFs (policies, HR forms, ISO docs) via Docling, chunks them with a tokenizer-aware `HybridChunker`, embeds with HuggingFace, and retrieves from a vector store.
- **Web search fallback** — Tavily search scoped to NextBridge's official domain, used only when internal docs don't have the answer.
- **Agentic tool use** — a LangGraph ReAct agent (not hardcoded if/else) decides which tool(s) to call: docs search, web search, or an action tool.
- **HR workflows via email** — employees can request leave, work-from-home, meal subscriptions, or file MIS complaints through natural conversation. The agent collects required fields, shows a confirmation summary, and only sends the email after explicit "yes."
- **Reply checking** — employees can ask for updates; the agent checks the inbox (IMAP) for department replies matching their employee ID, shows the reply, drafts an acknowledgment, and sends it after confirmation.
- **Conversation memory** — multi-turn flows (e.g., filling out a leave request over several messages) are remembered per chat session via a LangGraph checkpointer.
- **Streaming responses** — answers stream token-by-token to the frontend.
- **Text-to-speech** — a speaker button on each response reads it aloud instantly using the browser's native Web Speech API.
- **Speech-to-text** — a mic button lets employees speak their question instead of typing it.
- **Streamlit frontend** — multi-chat sidebar (like ChatGPT), new chat button, per-chat history.

## Tech Stack

| Layer | Tool |
|---|---|
| Document loading & chunking | Docling (`langchain-docling`, `HybridChunker`) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS |
| LLM | Groq (`llama-3.3-70b-versatile` / `openai/gpt-oss-20b`, swappable) |
| Agent framework | LangGraph (`create_react_agent`) |
| Web search | Tavily (domain-scoped) |
| Backend API | FastAPI |
| Email | Gmail SMTP (send) + IMAP (read replies) |
| Frontend | Streamlit |
| TTS | Browser-native Web Speech API |

## Project Structure

```
project_root/
├── main.py                        # CLI entrypoint (optional, for quick testing)
├── frontend/
│   └── app.py                     # Streamlit chat UI
├── app/
│   ├── main_api.py                 # FastAPI entrypoint
│   ├── src/
│   │   └── config.py               # All environment-driven settings
│   ├── api/
│   │   ├── routes.py                # /query, /health
│   │   ├── schemas.py               # Request/response models
│   │   ├── dependencies.py          # FastAPI Depends() helpers
│   │   └── state.py                 # Shared app state (vectorstore, etc.)
│   ├── ingestion/
│   │   ├── doclingLoader.py         # Load + chunk PDFs
│   │   └── run_ingestion.py         # Build or load the vector store
│   ├── embeddings/
│   │   └── embedder.py
│   ├── vectorstore/
│   │   └── faiss_store.py
│   ├── llm/
│   │   └── groq_llm.py
│   ├── agent/
│   │   └── nextbridge_agent.py      # System prompt, tool registry, agent builder
│   └── tools/
│       ├── rag_tool.py              # nextbridge_docs_search
│       ├── web_search.py            # nextbridge_web_search (Tavily)
│       ├── email_utils.py           # Shared SMTP send helper
│       ├── email_tool.py            # send_leave_email
│       ├── wfh_tool.py              # send_wfh_email
│       ├── meal_tool.py             # send_meal_subscription_email
│       ├── mis_tool.py              # send_mis_complaint_email
│       ├── email_reader.py          # IMAP search + parsing
│       ├── reply_tool.py            # check_pending_replies, get_reply_details
│       └── ack_tool.py              # send_acknowledgment_email
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key

SMTP_EMAIL=your_email@gmail.com
SMTP_APP_PASSWORD=your_16_char_app_password

GM_EMAIL=gm@nextbridge.com
FOOD_DEPT_EMAIL=food@nextbridge.com
MIS_EMAIL=mis@nextbridge.com
```

**Gmail App Password:** Google Account → Security → 2-Step Verification (must be on) → App Passwords → generate one for "Mail". Use that here, not your real Gmail password. IMAP must also be enabled: Gmail Settings → Forwarding and POP/IMAP → Enable IMAP.

### 3. Add source documents

Drop your PDFs (policies, handbooks, forms) into the folder set by `DATA_PATH` in `app/src/config.py`.

### 4. Run the backend

```bash
uvicorn main:app --reload
```

First run will parse, chunk, embed, and index your PDFs (can take a while depending on document count/size), then save the FAISS index locally so future runs load instantly.

### 5. Run the frontend

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

## Notes & Known Limitations

- **In-memory conversation checkpointer** — chat memory resets if the backend restarts. Swap `MemorySaver` for `SqliteSaver` (`langgraph-checkpoint-sqlite`) for persistence across restarts.
- **No database for requests/replies** — reply matching works by parsing the employee ID and request type directly out of email subject lines, not a stored record. This means a reply will keep showing as "available" every time it's checked, since nothing is marked as acknowledged.
- **Docling OCR is disabled** (`do_ocr = False`) since source PDFs are digital text + tables, not scanned images. Re-enable if any source documents are scanned.
- **Gmail SMTP/IMAP** is meant as a placeholder for testing. Swap for a dedicated provider (Resend, Brevo) before any real deployment.