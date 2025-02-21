import time

import psycopg2
from openai import OpenAI
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Database connection
DB_CONFIG = "dbname=virtualTeacher user=postgres password=password host=localhost"
conn = psycopg2.connect(DB_CONFIG)
cursor = conn.cursor()

api_key ="api_key"
# Load PhoBERT model & tokenizer
phobert = AutoModel.from_pretrained("vinai/phobert-large").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")

# Function to generate PhoBERT embeddings
def get_phobert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    tokens = {k: v.to("cuda") for k, v in tokens.items()}  # Move to GPU

    with torch.no_grad():
        output = phobert(**tokens)

    embedding = output.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling
    return embedding.flatten()

# Sample user data
username = "teacher1"
email = "teacher1@example.com"
cursor.execute("SELECT user_id FROM users WHERE email = %s", (email,))
user = cursor.fetchone()
if user is None:
    cursor.execute("""
        INSERT INTO users (username, email) 
        VALUES (%s, %s) 
        RETURNING user_id;
    """, (username, email))
    user_id = cursor.fetchone()[0]
    print(f"✅ User {username} created with user_id {user_id}")
else:
    user_id = user[0]
    print(f"✅ User {username} already exists with user_id {user_id}")

# Read questions from file
file_path = r"D:\ML\virtual_teacher\data\teacher_questions.txt"
with open(file_path, "r", encoding="utf-8") as f:
    questions = f.readlines()

# Process each question one by one
for idx, question in enumerate(questions[227:600]):  # Only process the first 600 questions
    question = question.strip()
    if not question:
        continue  # Skip empty lines

    print(f" Processing question {idx+1}: {question}")

    # Step 1: Insert Question into chat_history
    cursor.execute("""
        INSERT INTO chat_history (user_id, role, message) 
        VALUES (%s, %s, %s)
        RETURNING chat_id;
    """, (user_id, 'user', question))

    chat_id = cursor.fetchone()[0]  # Get generated chat_id

    # Step 2: Generate and Store PhoBERT Embedding
    embedding = get_phobert_embedding(question)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"  # Correct format for PGVector

    cursor.execute("""
        INSERT INTO chat_embeddings (chat_id, embedding) 
        VALUES (%s, %s::vector);
    """, (chat_id, embedding_str))

    # Step 3: Generate AI Response using OpenAI (ChatGPT)
    client = OpenAI(api_key=api_key)  # New OpenAI client

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )

    ai_response = response.choices[0].message.content  # Extract response

    # Step 4: Store AI Response in chat_history
    cursor.execute("""
        INSERT INTO chat_history (user_id, role, message) 
        VALUES (%s, %s, %s);
    """, (user_id, 'assistant', ai_response))

    # Commit after each insertion
    conn.commit()

    print(f"✅ Successfully stored question {idx+1}")

    # Optional: Delay to avoid API rate limits
    time.sleep(1)  # Adjust as needed

# Commit changes
conn.commit()
cursor.close()
conn.close()

print("All questions processed successfully!")
