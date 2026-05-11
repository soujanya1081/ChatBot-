import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Load Environment Variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

if not groq_api_key:
    st.error('Missing GROQ_API_KEY. Please add it to your .env file.')
    st.stop()

# 2. Initialize Model
llm = ChatGroq(groq_api_key=groq_api_key, model='llama-3.3-70b-versatile')

# 3. Define Prompt with History
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Remember previous conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

# 4. State Management for Memory
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# 5. Wrap Chain with Memory
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 6. Streamlit UI
st.title("Chatbot with Memory")

if "session_id" not in st.session_state:
    st.session_state.session_id = "default_user"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. Chat Input & Response Logic
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to UI
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Invoke Chain
    with st.spinner("Thinking..."):
        try:
            # Pass input as a dict, and session_id in config
            response = chain_with_memory.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            bot_reply = response.content
            
            # Save and display assistant response
            st.session_state.messages.append({'role': 'assistant', 'content': bot_reply})
            with st.chat_message("assistant"):
                st.markdown(bot_reply)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Clear History Button
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.store = {}
    st.rerun()
