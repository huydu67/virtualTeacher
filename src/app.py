import streamlit as st
import psycopg2
import openai
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from utils import get_phobert_embedding, search_similar_question, get_stored_response, store_reflection, \
    generate_chatgpt_response, store_new_question_response, search_similar_reflection

# Database connection
DB_CONFIG = "dbname=virtualTeacher user=postgres password=password host=localhost"

# Load PhoBERT model & tokenizer
phobert = AutoModel.from_pretrained("vinai/phobert-large").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")

# Reflection Categories
reflection_categories = [
    "Quản lý lớp học",
    "Tạo động lực cho học sinh",
    "Giải quyết xung đột",
    "Phương pháp giảng dạy",
    "Cách đánh giá học sinh"
]


# Streamlit Chat UI
st.title("🧑‍🏫 AI Teacher Chatbot")

user_input = st.text_input("Hỏi AI về giảng dạy:")
# Add mode selection
mode = st.radio("Chọn chế độ:", ["AI Reflection Chat", "Reflective Dialogue"])



if user_input:
    user_id = 5  # Using a default user for now

    # Generate embedding for the user question
    user_embedding = get_phobert_embedding(user_input)

    # Search for a similar past question
    similar_question, chat_id = search_similar_question(user_embedding)

    if similar_question:
        # Retrieve stored response
        ai_response = get_stored_response(chat_id)
        st.write(f"✅ **Câu hỏi tương tự đã có trong hệ thống:** {similar_question}")
    else:
        # Generate a new response
        ai_response = generate_chatgpt_response(user_input)
        # Store new question and response in database
        store_new_question_response(user_id, user_input, ai_response)

    st.write(f"🤖 **AI:** {ai_response}")

elif mode == "Reflective Dialogue":
    reflection_input = st.text_area("Nhập phản xạ giảng dạy của bạn:")
    category = st.selectbox("Chọn danh mục phản xạ:", reflection_categories)

    if st.button("Gửi phản xạ"):
        user_id = 5  # Using a default user for now

        # Generate embedding for the user reflection
        user_embedding = get_phobert_embedding(reflection_input)

        # Search for a similar past reflection
        similar_reflection, chat_id = search_similar_reflection(user_embedding)

        if similar_reflection:
            # Retrieve stored AI feedback
            ai_feedback = get_stored_response(chat_id)
            st.write(f"✅ **Câu hỏi tương tự đã có trong hệ thống:** {similar_reflection}")
        else:
            # Store reflection and get AI-generated self-reflection prompt
            chat_id, ai_prompt = store_reflection(user_id, reflection_input, category)
            st.write(f"🤔 **AI Reflection Prompt:** {ai_prompt}")

            response_input = st.text_area("Trả lời câu hỏi phản xạ:")

            if response_input and st.button("Nhận phản hồi AI"):
                ai_feedback = generate_chatgpt_response(
                    f"Đưa ra phản hồi cho câu trả lời của giáo viên: {response_input}")

                # Store AI feedback in database
                conn = psycopg2.connect(DB_CONFIG)
                cursor = conn.cursor()
                cursor.execute("""
                            INSERT INTO chat_history (user_id, role, message, reflection_category)
                            VALUES (%s, %s, %s, %s);
                        """, (user_id, 'assistant', ai_feedback, category))
                conn.commit()
                cursor.close()
                conn.close()

                st.write(f"📌 **AI Feedback:** {ai_feedback}")
