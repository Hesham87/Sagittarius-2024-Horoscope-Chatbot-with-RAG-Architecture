import time
from main import Sagittarius
import streamlit as st

st.set_page_config(page_title="The astrologer Bot")
with st.sidebar:
    st.title('Lets talk about the stars')

# Initialize bot with caching to avoid reloading on every interaction
@st.cache_resource
def load_bot():
    return Sagittarius()

bot = load_bot()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's unravel the stars"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about your future"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Consulting the stars..."):
            # Get response from RAG chain
            try:
                response = bot.rag_chain.invoke(prompt)
            except Exception as e:
                response = f"Sorry, I encountered an error: {str(e)}"
            
            # Stream the response
            for chunk in response.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)  # Simulate streaming
            
            message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})