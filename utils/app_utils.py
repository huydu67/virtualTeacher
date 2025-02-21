import streamlit as st
import psycopg2
import openai
import torch
import os
import sounddevice as sd
import numpy as np
import wave
import time
from transformers import AutoModel, AutoTokenizer
from faster_whisper import WhisperModel

api_key = "api_key"

# Database connection
DB_CONFIG = "dbname=virtualTeacher user=postgres password=password host=localhost"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load PhoBERT model & tokenizer
phobert = AutoModel.from_pretrained("vinai/phobert-large").to(device)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")

# Load Whisper Model
model_size = "small"
model = WhisperModel(model_size, device=device, compute_type="int8")

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)


def get_phobert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    tokens = {k: v.to(device) for k, v in tokens.items()}  # Move to GPU

    with torch.no_grad():
        output = phobert(**tokens)

    embedding = output.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling
    return embedding.flatten()


def search_similar_question(user_embedding):
    conn = psycopg2.connect(DB_CONFIG)
    cursor = conn.cursor()

    embedding_str = "[" + ",".join(map(str, user_embedding)) + "]"
    cursor.execute("""
        SELECT h.message, h.chat_id
        FROM chat_history h
        JOIN chat_embeddings e ON h.chat_id = e.chat_id
        WHERE h.role = 'user'
        ORDER BY e.embedding <-> %s::vector
        LIMIT 1;
    """, (embedding_str,))

    result = cursor.fetchone()
    print("Result: " + str(result))
    cursor.close()
    conn.close()

    if result:
        return result[0], result[1]  # Return question text and chat_id
    return None, None


def search_similar_reflection(user_embedding):
    """Retrieves the most similar past reflection using PGVector (without threshold)"""
    conn = psycopg2.connect(DB_CONFIG)
    cursor = conn.cursor()

    embedding_str = "[" + ",".join(map(str, user_embedding)) + "]"
    cursor.execute("""
        SELECT reflection_text, reflection_id
        FROM reflections
        ORDER BY embedding <-> %s::vector
        LIMIT 1;
    """, (embedding_str,))

    result = cursor.fetchone()
    cursor.close()
    conn.close()

    return result if result else (None, None)  # Always return the top match


def get_stored_response(chat_id):
    conn = psycopg2.connect(DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
            SELECT message FROM chat_history
            WHERE role = 'assistant' 
            AND user_id = (SELECT user_id FROM chat_history WHERE chat_id = %s)
            ORDER BY created_at ASC LIMIT 1;
        """, (chat_id,))

    response = cursor.fetchone()
    cursor.close()
    conn.close()

    return response[0] if response else None


def search_chat_history_feedback(user_embedding):
    """Search for similar AI-generated feedback in `chat_embeddings` using PGVector"""
    conn = psycopg2.connect(DB_CONFIG)
    cursor = conn.cursor()

    embedding_str = "[" + ",".join(map(str, user_embedding)) + "]"
    cursor.execute("""
        SELECT h.message 
        FROM chat_history h
        JOIN chat_embeddings e ON h.chat_id = e.chat_id
        WHERE h.role = 'assistant'
        ORDER BY e.embedding <-> %s::vector
        LIMIT 1;
    """, (embedding_str,))

    result = cursor.fetchone()
    cursor.close()
    conn.close()

    return result[0] if result else "Không có phản hồi phù hợp trong hệ thống."


def get_stored_ai_question(reflection_id):
    """Retrieves the AI-generated self-reflection question for a given reflection_id"""
    conn = psycopg2.connect(DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT message FROM chat_history
        WHERE reflection_category = (
            SELECT reflection_category FROM reflections WHERE reflection_id = %s
        ) AND role = 'assistant'
        ORDER BY created_at ASC LIMIT 1;
    """, (reflection_id,))

    question = cursor.fetchone()
    cursor.close()
    conn.close()

    return question[0] if question else "Không có câu hỏi phản xạ nào được lưu trữ."


def generate_chatgpt_response(prompt):
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Function to store a new question and response in the database
def store_new_question_response(user_id, question, ai_response):
    conn = psycopg2.connect(DB_CONFIG)
    cursor = conn.cursor()

    # Insert new question
    cursor.execute("""
        INSERT INTO chat_history (user_id, role, message)
        VALUES (%s, %s, %s) RETURNING chat_id;
    """, (user_id, 'user', question))

    chat_id = cursor.fetchone()[0]

    # Insert embedding
    embedding = get_phobert_embedding(question)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    cursor.execute("""
        INSERT INTO chat_embeddings (chat_id, embedding)
        VALUES (%s, %s::vector);
    """, (chat_id, embedding_str))

    # Insert AI response
    cursor.execute("""
        INSERT INTO chat_history (user_id, role, message)
        VALUES (%s, %s, %s);
    """, (user_id, 'assistant', ai_response))

    conn.commit()
    cursor.close()
    conn.close()
    return ai_response


def get_stored_feedback(reflection_id):
    """Retrieves stored AI feedback for a given reflection_id"""
    conn = psycopg2.connect(DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT message FROM chat_history
        WHERE reflection_category = (
            SELECT reflection_category FROM reflections WHERE reflection_id = %s
        ) AND role = 'assistant'
        ORDER BY created_at DESC LIMIT 1;
    """, (reflection_id,))

    feedback = cursor.fetchone()
    cursor.close()
    conn.close()

    return feedback[0] if feedback else "Không có phản hồi nào được lưu trữ."


def store_new_reflection(user_id, reflection_text, category):
    """Stores a new reflection in the `reflections` table and embeds it in PGVector"""
    conn = psycopg2.connect(DB_CONFIG)
    cursor = conn.cursor()

    # Generate PhoBERT embedding
    embedding = get_phobert_embedding(reflection_text)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    cursor.execute("""
        INSERT INTO reflections (user_id, reflection_text, reflection_category, embedding)
        VALUES (%s, %s, %s, %s::vector);
    """, (user_id, reflection_text, category, embedding_str))

    conn.commit()
    cursor.close()
    conn.close()


def record_audio(filename, duration=30):
    """Record audio and save it to a file."""
    global audio_buffer
    file_path = os.path.join(DATA_FOLDER, filename)

    audio_buffer = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=np.int16, callback=callback):
        start_time = time.time()
        while st.session_state.recording and time.time() - start_time < duration:
            time.sleep(0.1)

    if audio_buffer:
        audio_data = np.concatenate(audio_buffer, axis=0)
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        st.session_state.audio_file = file_path

    st.session_state.recording = False
    st.rerun()


def transcribe_audio(file_path):
    """Transcribe recorded audio using Whisper."""
    segments, info = model.transcribe(file_path, beam_size=5, word_timestamps=True, vad_filter=True)
    transcribed_text = " ".join(segment.text for segment in segments)
    return transcribed_text

def stop_recording():
    """Stop the recording early."""
    global recording
    recording = False
    st.write("🛑 Recording stopped.")


def callback(indata, frames, time, status):
    if status:
        print(status)
    global audio_buffer
    audio_buffer.append(indata.copy())