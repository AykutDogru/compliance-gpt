import streamlit as st
from ollama_starter import OllamaStarter  # Import the OllamaStarter class

def main():
    st.title("Compliance GPT")

    # Initialize session state for vector database status
    if 'vector_database_exists' not in st.session_state:
        st.session_state.vector_database_exists = False

    # Sidebar for model and collection selection
    st.sidebar.header("Settings")
    
    model_names = ["qwen:0.5b","qwen2:1.5b", "llama3.1:8b", "llama3.1:70b"] 
    collection_names = ["mastercard", "visa", "amex", "troy"]  

    selected_model = st.sidebar.selectbox("Select Model", model_names)
    selected_collection = st.sidebar.selectbox("Select Payment Processor", collection_names)
    # Sidebar parameters for question answering
    st.sidebar.header("Model Parameters")
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=100, value=5)   
    top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.9)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7)

    # Sidebar for PDF file uploader
    st.sidebar.header("Upload PDF Files")
    uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    ollama_starter = None  # Initialize to None
    ollama_starter = OllamaStarter(selected_model,top_p,top_k,temperature, selected_collection, uploaded_files, "")
    st.session_state.vector_database_exists, message = ollama_starter.connect_existing_vector_store()
    st.write(message)

    if st.sidebar.button("Save Vector Database"):
        if selected_model and selected_collection and uploaded_files:
            ollama_starter = OllamaStarter(selected_model,top_p,top_k,temperature, selected_collection, uploaded_files, "")
            message, vectorstore = ollama_starter.process()  # Get process message and vectorstore
            st.success(message)
            if vectorstore:
                summary = ollama_starter.get_document_summary()  # Get the summary
                st.write(summary)
                st.session_state.vector_database_exists = True  # Mark as existing
        else:
            st.error("Please select a model, payment processor, and upload files.")

    # Main page question input
    st.header("Ask a Question")
    question = st.text_area("Enter your question here:")

    if st.button("Submit Question"):
        if question:
            if st.session_state.vector_database_exists:
                if ollama_starter:
                    # Pass top_k, top_p, and temperature to the answer_question method
                    response = ollama_starter.answer_question(question)
                    st.write("Response:", response)
                else:
                    st.error("Please save the vector database first.")
            else:
                st.error("Vector database does not exist. Please upload files and save the database first.")
        else:
            st.error("Please enter a question.")

if __name__ == "__main__":
    main()
