import streamlit as st
import asyncio
import os
from app2 import JioPayChatbot  # Import from app2.py

# Set up the app
st.set_page_config(page_title="JioPay Assistant", page_icon="🤖")

def initialize_session_state():
    """Initialize session state variables"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def setup_chatbot():
    """Initialize the JioPay chatbot"""
    try:
        chatbot = JioPayChatbot()
        chatbot.create_knowledge_base()
        chatbot.initialize_qa()
        return chatbot
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None

def main():
    initialize_session_state()
    
    st.title("JioPay Customer Support Assistant")
    st.markdown("Ask me anything about JioPay services!")

    # Initialize chatbot once
    if not st.session_state.initialized:
        with st.spinner("Initializing assistant... (this may take a minute)"):
            st.session_state.chatbot = setup_chatbot()
            if st.session_state.chatbot:
                st.session_state.initialized = True
                if not st.session_state.messages:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Welcome to JioPay Support! How can I help you today?"
                    })

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View Sources"):
                    st.write(message["sources"])

    # Chat interface
    if st.session_state.initialized:
        user_input = st.chat_input("Ask your question about JioPay...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message immediately
            with st.chat_message("user"):
                st.write(user_input)
            
            # Show thinking indicator for assistant
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get assistant response
                    response = st.session_state.chatbot.ask(user_input)
                    
                    # Process response for sources
                    if "Sources:" in response:
                        answer, sources = response.split("Sources:", 1)
                        answer = answer.strip()
                        sources = sources.strip()
                    else:
                        answer = response
                        sources = None
                    
                    # Add assistant response to chat history
                    assistant_message = {"role": "assistant", "content": answer}
                    if sources:
                        assistant_message["sources"] = sources
                    
                    st.write(answer)
                    if sources:
                        with st.expander("View Sources"):
                            st.write(sources)
            
            # Add the message to session state
            st.session_state.messages.append(assistant_message)

if __name__ == "__main__":
    main()