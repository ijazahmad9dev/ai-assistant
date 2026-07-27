import uuid
import requests
import streamlit as st

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="NextBridge Assistant", layout="wide")

# ---- Initialize state ----
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None


def new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"title": "New Chat", "messages": []}
    st.session_state.active_chat_id = chat_id


# Start with one chat if none exist yet
if not st.session_state.chats:
    new_chat()


# ---- Sidebar ----
with st.sidebar:
    st.title("NextBridge Assistant")

    if st.button("➕ New Chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()
    st.caption("Your chats")

    # Show most recent chats first
    for chat_id in reversed(list(st.session_state.chats.keys())):
        chat = st.session_state.chats[chat_id]
        is_active = chat_id == st.session_state.active_chat_id
        label = ("🟢 " if is_active else "") + chat["title"]

        if st.button(label, key=f"chat_{chat_id}", use_container_width=True):
            st.session_state.active_chat_id = chat_id
            st.rerun()


# ---- Main chat area ----
active_id = st.session_state.active_chat_id
active_chat = st.session_state.chats[active_id]

st.header(active_chat["title"])

# Display existing messages
for msg in active_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask NextBridge Assistant...")

if user_input:
    # Show user message immediately
    active_chat["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Auto-title the chat from the first message
    if active_chat["title"] == "New Chat":
        active_chat["title"] = user_input[:40] + ("..." if len(user_input) > 40 else "")

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": user_input, "session_id": active_id},
                    timeout=60,
                )
                response.raise_for_status()
                answer = response.json()["answer"]
            except Exception as e:
                answer = f"Error contacting the assistant: {e}"

        st.markdown(answer)

    active_chat["messages"].append({"role": "assistant", "content": answer})
    st.rerun()