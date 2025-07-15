import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.title("📚 LangGraph Research Chatbot")

user_input = st.text_input("Ask a question about AI/ML research")

if st.button("Send") and user_input.strip():
    try:
        response = requests.post(API_URL, json={"message": user_input})
        response.raise_for_status()
        st.markdown("### 🤖 Response")
        st.write(response.json()["response"])
    except Exception as e:
        st.error(f"Error: {e}")
