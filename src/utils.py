import streamlit as st
import psycopg2
import openai
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

api_key = "api-key"
# Database connection
DB_CONFIG = "dbname=virtualTeacher user=postgres password=password host=localhost"

# Load PhoBERT model & tokenizer
phobert = AutoModel.from_pretrained("vinai/phobert-large").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")


def get_phobert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    tokens = {k: v.to("cuda") for k, v in tokens.items()}  # Move to GPU

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


def search_similar_reflection(user_embedding, threshold=0.2):
    conn = psycopg2.connect(DB_CONFIG)
    cursor = conn.cursor()

    embedding_str = "[" + ",".join(map(str, user_embedding)) + "]"
    cursor.execute("""
        SELECT h.message, h.chat_id, e.embedding <-> %s::vector AS distance
        FROM chat_history h
        JOIN chat_embeddings e ON h.chat_id = e.chat_id
        WHERE h.role = 'user'
        ORDER BY distance ASC
        LIMIT 1;
    """, (embedding_str,))

    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result and result[2] < threshold:  # Check if similarity score is below threshold
        return result[0], result[1]  # Return reflection text and chat_id
    return None, None


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


# Function to store teacher reflection and update category
def store_reflection(user_id, reflection, category):
    conn = psycopg2.connect(DB_CONFIG)
    cursor = conn.cursor()

    # Store reflection in chat_history with category
    cursor.execute("""
        INSERT INTO chat_history (user_id, role, message, reflection_category)
        VALUES (%s, %s, %s, %s) RETURNING chat_id;
    """, (user_id, 'user', reflection, category))

    chat_id = cursor.fetchone()[0]

    # Generate AI self-reflection question
    ai_prompt = generate_chatgpt_response(f"Hãy đặt câu hỏi phản xạ cho giáo viên: {reflection}")

    # Store AI reflection prompt
    cursor.execute("""
        INSERT INTO chat_history (user_id, role, message, reflection_category)
        VALUES (%s, %s, %s, %s);
    """, (user_id, 'assistant', ai_prompt, category))

    conn.commit()
    cursor.close()
    conn.close()
    return chat_id, ai_prompt


# Function to generate reflection prompt
def generate_reflection_prompt(reflection):
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Giả sử bạn là một AI hỗ trợ giáo viên tự đánh giá. Hãy giúp tôi phản xạ về vấn đề sau: {reflection}"}
        ]
    )
    return response.choices[0].message.content


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
