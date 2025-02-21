import streamlit as st
import torch
import os
import sys
import sounddevice as sd
import numpy as np
import wave
import time
from transformers import AutoModel, AutoTokenizer

# Get the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import app_utils

# Database connection
DB_CONFIG = "dbname=virtualTeacher user=postgres password=password host=localhost"
device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Load PhoBERT model & tokenizer
phobert = AutoModel.from_pretrained("vinai/phobert-large").to(device)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
#
# # Reflection Categories
# reflection_categories = [
#     "Quản lý lớp học",
#     "Tạo động lực cho học sinh",
#     "Giải quyết xung đột",
#     "Phương pháp giảng dạy",
#     "Cách đánh giá học sinh",
#     "Động lực học tập"
# ]


# Streamlit Chat UI

if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None

st.set_page_config(page_title="AI Personal Assistant")
st.title("🧑‍🏫 AI Teacher Chatbot")

# Chat history for UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False  # Track if waiting for user response

if not st.session_state.recording:
    if st.button("🎤 Start Recording"):
        st.session_state.recording = True
        st.session_state.audio_file = None
        st.rerun()
else:
    if st.button("🛑 Stop Recording"):
        st.session_state.recording = False
        st.rerun()

if st.session_state.recording:
    st.write("🎙️ Recording... Speak now!")

uploaded_file = st.file_uploader("Upload a voice file (.mp3 or .wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.audio_file = file_path
    transcribed_text = app_utils.transcribe_audio(file_path)
    if transcribed_text:
        st.write(f"📝 Transcribed Text: {transcribed_text}")
        st.session_state.chat_history.append({"role": "user", "content": transcribed_text})

if not st.session_state.recording and st.session_state.audio_file:
    transcribed_text = app_utils.transcribe_audio(st.session_state.audio_file)
    if transcribed_text:
        st.write(f"📝 Transcribed Text: {transcribed_text}")
        st.session_state.chat_history.append({"role": "user", "content": transcribed_text})
        st.session_state.audio_file = None

# Display chat history
st.write("### 📩 Lịch sử hội thoại")
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User text input
reflection_input = st.text_input("Nhập vấn đề giảng dạy của bạn:", key="reflection_input") if not st.session_state.awaiting_response else None

if reflection_input:
    user_id = 5  # Default user ID

    # Generate embedding for the user reflection
    user_embedding = app_utils.get_phobert_embedding(reflection_input)

    # Search for a similar past reflection using PGVector
    similar_reflection, reflection_id = app_utils.search_similar_reflection(user_embedding)

    if similar_reflection:
        # Retrieve stored AI-generated self-reflection question
        ai_question = app_utils.get_stored_ai_question(reflection_id)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"✅ **Câu hỏi tương tự đã có trong hệ thống:** {similar_reflection}"})
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"🤔 **AI Reflection Prompt:** {ai_question}"})
    else:
        # Store new reflection if no match is found
        app_utils.store_new_reflection(user_id, reflection_input)
        st.session_state.chat_history.append({"role": "assistant",
                                              "content": "✅ **Phản xạ của bạn đã được lưu trữ.** Hãy tiếp tục làm việc với phản xạ này trong tương lai."})

    # Mark that the chatbot is awaiting a response from the user
    st.session_state.awaiting_response = True
    reflection_input = ""  # Clear input after submission


# Handle user response input (Only enabled after a reflection question is shown)
if st.session_state.awaiting_response:
    response_input = st.text_input("Nhập vấn đề giảng dạy của bạn:", key="response_input")

    if response_input:
        # Generate embedding for user response
        response_embedding = app_utils.get_phobert_embedding(response_input)

        # Search `chat_history` for the most relevant AI-generated feedback
        similar_feedback = app_utils.search_chat_history_feedback(response_embedding)

        # Store user response in chat history
        st.session_state.chat_history.append({"role": "user", "content": response_input})

        if similar_feedback:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"📌 **AI Feedback:** {similar_feedback}"})
        else:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "Không có phản hồi phù hợp trong hệ thống. Hãy thử lại sau!"})

        # Reset the flag to allow new reflections to be entered
        st.session_state.awaiting_response = False





