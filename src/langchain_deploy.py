import streamlit as st
from utils.models.langchain_model import LangChainChatBot


chatbot = LangChainChatBot()
chatbot.read_data()
chatbot.preprocess()
query_engine = chatbot.generate_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []
st.title("Let's try!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("質問を入力してください。"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("回答を生成中..."):
        response = query_engine.query(prompt)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
