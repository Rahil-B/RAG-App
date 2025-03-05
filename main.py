
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Your Personal Social Media Assistant",
    page_icon="🚀",
    layout="wide",
)

from utils.embedding_utils import (
    load_embedding_model, 
    load_sentence_transformer, 
    add_documents_to_collection,
    query_chroma
)
from utils.web_page import fetch_data_from_urls
from utils.llm_utils import (
    detect_hallucination, generate_response, groundness_func, answer_relevance_func, 
    context_relevance_func, cosine_similarity_func, perplexity_score
)
from utils.analysis_utils import (
    Neutrality_viz, answer_relevance, calculate_toxicity, calculate_sentiment, context_relevance, rouge_score, 
    bias_score_func, detect_pii, visualize_groundness, visualize_toxicity
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


# Custom CSS for better button alignment
st.markdown(
    """
    <style>
        /* Center the content */
        .main-content {
            max-width: 800px;
            margin: auto;
            padding-top: 40px;
        }

        /* Heading style */
        .title {
            font-size: 2.3rem;
            font-weight: bold;
        }

        /* Subtext */
        .subtext {
            font-size: 1.1rem;
            color: #bbb;
        }

        /* Button container - flexbox for better alignment */
        .button-container {
            display: flex;
            justify-content: left;
            gap: 15px; /* Adjust spacing between buttons */
            margin-top: 20px;
        }

        /* Button styling */
        .stButton button {
            width: 180px;
            height: 45px;
            font-size: 1rem;
            font-weight: bold;
            border-radius: 8px;
            background-color: #007BFF;
            color: white;
            border: none;
        }

        .stButton button:hover {
            background-color: #0056b3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main container for content
with st.container():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # Title
    st.markdown('<div class="title">🚀 Welcome to Your Personal Social Media Assistant!</div>', unsafe_allow_html=True)

    # Description
    st.markdown(
        """
        <div class="subtext">
        We’re here to help you create eye-catching, engaging posts in no time.  
        Whether you're looking to share news, promote a product, or simply interact with your audience, we've got you covered!
        <br><br>
        👉 <b>What would you like to do today?</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Buttons inside a flexbox container
    st.markdown('<div class="button-container">', unsafe_allow_html=True)

    col1, col2 = st.columns([0.2, 0.8])  # Reduce spacing between buttons
    with col1:
        if st.button("📤 Upload Data"):
            st.switch_page("pages/data-upload.py")
    with col2:
        if st.button("✍️ Create Post"):
            st.switch_page("pages/create-post.py")

    st.markdown('</div>', unsafe_allow_html=True)