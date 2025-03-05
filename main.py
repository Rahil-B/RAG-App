
import streamlit as st
st.set_page_config(
    page_title="Responsible AI RAG System",
    page_icon="🤖",
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
    generate_response, groundness_func, answer_relevance_func, 
    context_relevance_func, cosine_similarity_func, perplexity_score
)


from utils.analysis_utils import (
    calculate_toxicity, calculate_sentiment, rouge_score, 
    bias_score_func, detect_pii, visualize_toxicity
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

if st.button("Create post"):
    st.switch_page("pages/create-post.py")
elif st.button("Data Upload"):
    st.switch_page("pages/data-upload.py")
# # main.py
# import streamlit as st
# from utils.pdf_utils import extract_text_from_pdf, chunk_text
# from utils.reddit_utils import fetch_reddit_posts
# from utils.embedding_utils import load_embedding_model, load_sentence_transformer, add_documents_to_collection, query_chroma
# from utils.llm_utils import generate_response, groundness_func, answer_relevance_func, context_relevance_func, cosine_similarity_func, perplexity_score
# from utils.analysis_utils import calculate_toxicity, calculate_sentiment, rouge_score, bias_score_func, detect_pii, visualize_toxicity

# st.title("Responsible AI RAG System")

# # Initialize cached embedding models
# embedding_model = load_embedding_model()
# sentence_model = load_sentence_transformer()

# # Upload Documents or Provide Subreddit Name
# option = st.radio("Choose Input Type", ["Upload Document", "Enter Subreddit Name"])

# if option == "Upload Document":
#     uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])    
#     if uploaded_file is not None:
#         with st.spinner("Processing... Please wait ⏳"):              
#             raw_text = extract_text_from_pdf(uploaded_file)        
#             text_chunks = chunk_text(raw_text)
#             if text_chunks:
#                 add_documents_to_collection(text_chunks, sentence_model)
#                 st.success("Document uploaded and stored in ChromaDB!")
#             else:
#                 st.error("No text extracted from PDF. Try another document.")

# elif option == "Enter Subreddit Name":
#     subreddit_name = st.text_input("Enter Subreddit Name")
#     if st.button("Fetch Reddit Data"):
#         with st.spinner("Fetching... Please wait ⏳"):
#             raw_text = "\n".join(fetch_reddit_posts(subreddit_name, limit=10))
#             text_chunks = chunk_text(raw_text)
#             if text_chunks:
#                 add_documents_to_collection(text_chunks, sentence_model)
#                 st.success("Reddit posts uploaded and stored in ChromaDB!")
#             else:
#                 st.error("No text extracted from Reddit. Try again.")

# # Initialize session state for answer generation
# if "answer_generated" not in st.session_state:
#     st.session_state.answer_generated = False

# # Query Input
# st.header("Ask a Question")
# question = st.text_input("Enter your question")

# if st.button("Get Answer"):
#     st.session_state.answer_generated = True

# if st.session_state.answer_generated:
#     with st.spinner("Processing... Please wait ⏳"):
#         relevant_docs = query_chroma(question, embedding_model)
#         context = "\n".join(relevant_docs) if relevant_docs else "No relevant context found."
#         response = generate_response(question, context)
        
#         st.subheader("Generated Response")
#         st.write(response)
        
#         # Responsible AI Analysis
#         toxicity_score = calculate_toxicity(response)
#         sentiment_scores = calculate_sentiment(response)
#         groundness_score = groundness_func(context, response)
#         answer_relevance_score = answer_relevance_func(response, question)
#         context_relevance_score = context_relevance_func(context, question)
#         rouge_scores = rouge_score(response, context)
#         perplexity_scores = perplexity_score(response)
#         bias_score = bias_score_func(response)
#         cosine_similarity_score = cosine_similarity_func(question, response)
#         pii = detect_pii(response)
        
#         st.subheader("Responsible AI Analysis")
#         visualize_toxicity(toxicity_score)
#         st.write(f"**Toxicity Score:** {toxicity_score}")
#         st.write(f"**Hallucination:** {'No Hallucination' if cosine_similarity_score > 0.4 else 'Hallucinated'}")
#         st.write(f"**Cosine Similarity Score:** {cosine_similarity_score}")        
#         st.write(f"**Groundness Score:** {groundness_score}")
#         st.write(f"**Answer Relevance Score:** {answer_relevance_score}")
#         st.write(f"**Context Relevance Score:** {context_relevance_score}")
#         st.write(f"**Neutrality:** {sentiment_scores['neutrality']}")
#         st.write(f"**Subjectivity:** {sentiment_scores['subjectivity']}")
#         st.write(f"**Polarity:** {sentiment_scores['polarity']}")
#         st.write(f"**PII Detection:** {pii}")
#         st.write(f"**ROUGE Score:** {rouge_scores}")        
#         st.write(f"**Perplexity Score:** {perplexity_scores}")
#         st.write(f"**Bias Score:** {bias_score}")
