import streamlit as st
st.set_page_config(
    page_title="Data Upload",
    page_icon="📂",
    layout="wide",
)
from utils.pdf_utils import extract_text_from_pdf, chunk_text
from utils.reddit_utils import fetch_reddit_posts
from utils.embedding_utils import (
    load_embedding_model, 
    load_sentence_transformer, 
    add_documents_to_collection,
    query_chroma
)
from utils.web_page import fetch_data_from_urls
sentence_model = load_sentence_transformer()
st.header("Data Upload")
option = st.radio("Choose Input Type", ["Upload Document", "Fetch Data from Reddit", "Fetch Data from URL"])

if option == "Upload Document":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Processing... Please wait ⏳"):
            raw_text = extract_text_from_pdf(uploaded_file)
            text_chunks = chunk_text(raw_text)
            if text_chunks:
                add_documents_to_collection(text_chunks, sentence_model)
                st.success("Document uploaded and stored in ChromaDB!")
            else:
                st.error("No text extracted from PDF. Try another document.")

elif option == "Fetch Data from Reddit":
    subreddit_name = st.text_input("Enter Subreddit Name")
    if st.button("Fetch Reddit Data"):
        with st.spinner("Fetching... Please wait ⏳"):
            raw_text = "\n".join(fetch_reddit_posts(subreddit_name, limit=10))
            text_chunks = chunk_text(raw_text)
            if text_chunks:
                add_documents_to_collection(text_chunks, sentence_model)
                st.success("Reddit posts uploaded and stored in ChromaDB!")
            else:
                st.error("No text extracted from Reddit. Try again.")

elif option == "Fetch Data from URL":
    # Allow user to input up to 3 URLs
    urls = [st.text_input(f"Enter URL {i+1}") for i in range(3)]
    if st.button("Fetch Data from URLs"):
        with st.spinner("Fetching... Please wait ⏳"):
            docs_list = fetch_data_from_urls(urls)
            raw_text = "\n".join(docs_list)
            text_chunks = chunk_text(raw_text)
            if text_chunks:
                add_documents_to_collection(text_chunks, sentence_model)
                st.success("Data from URLs uploaded and stored in ChromaDB!")
            else:
                st.error("No text extracted from URLs. Try again.")

# Button to go back to "Ask a Question"
if st.button("Proceed to Ask a Question"):
    st.switch_page("pages/create-post.py")
