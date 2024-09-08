import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import textwrap
from html_templates import css, user_template, bot_template
api = st.secrets["GOOGLE_API_KEY"]
# Wrapping the text for better display
def wrap_text(text, width=90):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

# Extracting PDF text from uploaded files
def get_pdf_text(pdf_docs):
    docs = []
    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf.read())
            temp_file_path = temp_file.name
        loader = PyPDFLoader(temp_file_path)
        docs.extend(loader.load())
        os.remove(temp_file_path)
    return docs

# Create vector store for document embeddings
def get_vectorstore(docs):
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key = api)
    vector_store = Chroma.from_documents(docs, gemini_embeddings)
    return vector_store

# Generate a conversation chain
def get_conversation_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key = api)
    
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
    response = st.session_state.conversation.invoke(user_question)
    st.session_state.chat_history.append({'role': 'user', 'content': user_question})
    st.session_state.chat_history.append({'role': 'bot', 'content': wrap_text(response)})

    # Display the conversation history (normal order for a chat-like appearance)
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)

# Main function to create the layout
def main():
    st.set_page_config(page_title="AccountAssistant", page_icon="🏦")
    load_dotenv()
    st.write(css, unsafe_allow_html=True)

    # Initialize conversation and chat history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    # Move the header to the top of the page
    st.header("Account Assistant")
    


    # PDF uploading and processing
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your Bank Statement as PDF and click Process", accept_multiple_files=True)
        
        # If the "Process" button is clicked
        if st.button("Process"):
            # Clear all previous conversation, chat history, and vector store
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.vector_store = None

            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    docs = get_pdf_text(pdf_docs)
                    
                    # Reset and create a new vector store with the new documents
                    vector_store = get_vectorstore(docs)
                    st.session_state.vector_store = vector_store
                    
                    # Create a new conversation chain
                    st.session_state.conversation = get_conversation_chain(st.session_state.vector_store)
                    st.success("Bank Statement processed successfully!")
            else:
                st.error("Please upload a Bank Statement PDF file.")

    # Display the conversation history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)

    # Move the input box to the bottom of the page
    user_question = st.chat_input("Type your question here...", key="user_question")
    
    # Handle the user input when submitted
    if user_question:
        handle_user_input(user_question)

# Run the main function
if __name__ == "__main__":
    main()
