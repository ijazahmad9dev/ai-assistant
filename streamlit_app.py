import uuid
import requests
import streamlit as st
import os
from streamlit_mic_recorder import speech_to_text
import streamlit.components.v1 as components


def speak_button(text: str, key: str):
    """Renders a mic/speaker button that uses the browser's native TTS —
    starts speaking immediately, no network call, no generation delay."""
    safe_text = text.replace("\\", "\\\\").replace("`", "\\`").replace("\n", " ")

    html_code = f"""
    <button id="speak-btn-{key}" style="
        background: none;
        border: none;
        cursor: pointer;
        font-size: 20px;
    ">🔊</button>

    <script>
        const btn = document.getElementById("speak-btn-{key}");
        btn.addEventListener("click", () => {{
            window.speechSynthesis.cancel();  // stop any currently playing speech
            const utterance = new SpeechSynthesisUtterance(`{safe_text}`);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }});
    </script>
    """
    components.html(html_code, height=40)

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

for i, msg in enumerate(active_chat["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            speak_button(msg["content"], key=f"{active_id}_{i}")

# Chat input
input_col, mic_col = st.columns([0.94, 0.06])

with mic_col:
    voice_text = speech_to_text(
        language="en",
        start_prompt="🎤",
        stop_prompt="⏹️",
        just_once=True,
        use_container_width=True,
        key=f"mic_{active_id}",
    )

with input_col:
    typed_text = st.chat_input("Ask NextBridge Assistant...")

user_input = typed_text or voice_text

if user_input:
    active_chat["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if active_chat["title"] == "New Chat":
        active_chat["title"] = user_input[:40] + ("..." if len(user_input) > 40 else "")

    with st.chat_message("assistant"):
        def stream_response():
            try:
                with requests.post(
                    API_URL,
                    json={"question": user_input, "session_id": active_id},
                    stream=True,
                    timeout=120,
                ) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            yield chunk
            except Exception as e:
                yield f"Error contacting the assistant: {e}"

        full_response = st.write_stream(stream_response())

    active_chat["messages"].append({"role": "assistant", "content": full_response})
    st.rerun()