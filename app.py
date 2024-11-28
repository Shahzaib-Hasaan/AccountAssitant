import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
import textwrap
from html_templates import css, user_template, bot_template

load_dotenv()

def wrap_text(text, width=90):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def get_webpage_text(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

def get_vectorstore(docs):
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, gemini_embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Handle user input and display chat history
def handle_user_input(user_question):
    # Check for friendly response
    friendly_response = get_friendly_response(user_question)
    if friendly_response:
        response = friendly_response
    else:
        try:
            response = st.session_state.conversation.invoke(user_question)
            if not response.strip():
                response = "I'm here to assist you! Let me know if you have more questions."
        except Exception as e:
            response = "I'm sorry, I couldn't understand that. Please try rephrasing your question."

    # Append to chat history
    st.session_state.chat_history.append({'role': 'user', 'content': user_question})
    st.session_state.chat_history.append({'role': 'bot', 'content': wrap_text(response)})

    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)


friendly_responses = {
    "thank you": "You're welcome! Let me know if you need more help.",
    "bye": "Goodbye! Feel free to return anytime.",
    "thanks": "You're welcome! Let me know if you need more help.",
    "goodbye": "Goodbye! Feel free to return anytime.",
    "ok": "Alright! Let me know if you have more questions.",
    "okay": "Alright! Let me know if you have more questions.",
    "Ok Thanks": "You're welcome! Let me know if you need more help.",
    "Okay Thanks": "You're welcome! Let me know if you need more help.",
    "Ok thank you": "You're welcome! Let me know if you need more help.",
}

def get_friendly_response(input_text):
    input_text = input_text.lower().strip()
    for key, response in friendly_responses.items():
        if key in input_text:
            return response
    return None


def main():
    st.set_page_config(page_title="Chat with webpage", page_icon="🔗")
    load_dotenv()
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    # st.header("Account Assistant")

    with st.sidebar:
        st.subheader("Input Webpage URL")
        webpage_url = st.text_input("Paste the URL of the webpage and click Process")
        
        if st.button("Process"):
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.vector_store = None

            if webpage_url:
                with st.spinner("Processing the webpage..."):
                    try:
                        docs = get_webpage_text(webpage_url)
                        vector_store = get_vectorstore(docs)
                        st.session_state.vector_store = vector_store
                        st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
                        st.success("Webpage processed successfully!")
                    except Exception as e:
                        st.error(f"Failed to process the webpage: {e}")
            else:
                st.error("Please enter a valid URL.")

    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)

    user_question = st.chat_input("Type your question here...", key="user_question")
    
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()
