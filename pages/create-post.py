import streamlit as st

st.set_page_config(
    page_title="Create Post",
    page_icon="❓",
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
embedding_model = load_embedding_model()

# st.header("Data Check")
# has_data = st.radio("Do you have any data you wish to upload?", ["No", "Yes"])

# if has_data == "Yes":
#     st.info("Redirecting you to the Data Upload page...")
#     st.switch_page("pages/data-upload.py")  # Navigate to Data Upload
# else:

st.header("Ask a Question")
question = st.text_input("Enter your question", key="question_input")
    
if st.button("Get Answer"):
        st.session_state.answer_generated = True

if st.session_state.get("answer_generated") and question:
        with st.spinner("Processing... Please wait ⏳"):
            relevant_docs = query_chroma(question, embedding_model)
            context = "\n".join(relevant_docs) if relevant_docs else "No relevant context found."
            response = generate_response(question, context)
            
            st.subheader("Generated Response")
            st.write(response)
            
            # # Responsible AI Analysis
            # toxicity_score = calculate_toxicity(response)
            # sentiment_scores = calculate_sentiment(response)
            # groundness_score = groundness_func(context, response)
            # answer_relevance_score = answer_relevance_func(response, question)
            # context_relevance_score = context_relevance_func(context, question)
            # rouge_scores = rouge_score(response, context)
            # perplexity_scores = perplexity_score(response)
            # bias_score = bias_score_func(response)
            # cosine_similarity_score = cosine_similarity_func(question, response)
            # pii = detect_pii(response)
            
            # st.subheader("Responsible AI Analysis")
            # visualize_toxicity(toxicity_score)
            # st.write(f"**Toxicity Score:** {toxicity_score}")
            # st.write(f"**Hallucination:** {'No Hallucination' if cosine_similarity_score > 0.4 else 'Hallucinated'}")
            # st.write(f"**Cosine Similarity Score:** {cosine_similarity_score}")        
            # st.write(f"**Groundness Score:** {groundness_score}")
            # st.write(f"**Answer Relevance Score:** {answer_relevance_score}")
            # st.write(f"**Context Relevance Score:** {context_relevance_score}")
            # st.write(f"**Neutrality:** {sentiment_scores.get('neutrality', 'N/A')}")
            # st.write(f"**Subjectivity:** {sentiment_scores.get('subjectivity', 'N/A')}")
            # st.write(f"**Polarity:** {sentiment_scores.get('polarity', 'N/A')}")
            # st.write(f"**PII Detection:** {pii}")
            # st.write(f"**ROUGE Score:** {rouge_scores}")        
            # st.write(f"**Perplexity Score:** {perplexity_scores}")
            # st.write(f"**Bias Score:** {bias_score}")

            # Responsible AI Analysis
            toxicity_score = calculate_toxicity(response)
            sentiment_scores = calculate_sentiment(response)
            groundness_score = groundness_func(context, response)
            answer_relevance_score = answer_relevance_func(response, question)
            context_relevance_score = context_relevance_func(context, question)
            rouge_scores = rouge_score(response, context)
            perplexity_scores = perplexity_score(response)
            bias_score = bias_score_func(response)
            cosine_similarity_score = cosine_similarity_func(question, response)
            pii = detect_pii(response)
            
            st.header("Responsible AI Analysis")
            visualize_toxicity(toxicity_score)
            st.write(f"**Toxicity Score:** {toxicity_score}")
            # st.write(f"**Hallucination:** {'No Hallucination' if cosine_similarity_score > 0.4 else 'Hallucinated'}")
            detect_hallucination(question, response)
            st.write(f"**Cosine Similarity Score:** {cosine_similarity_score}")  
            visualize_groundness(groundness_score)      
            st.write(f"**Groundness Score:** {groundness_score}")
            answer_relevance(answer_relevance_score)
            st.write(f"**Answer Relevance Score:** {answer_relevance_score}")
            context_relevance(context_relevance_score)
            st.write(f"**Context Relevance Score:** {context_relevance_score}")
            Neutrality_viz(sentiment_scores['neutrality'])
            st.write(f"**Neutrality:** {sentiment_scores['neutrality']}")
            st.write(f"**Subjectivity:** {sentiment_scores['subjectivity']}")
            st.write(f"**Polarity:** {sentiment_scores['polarity']}")
            st.write(f"**PII Detection:** {pii}")
            st.write(f"**ROUGE Score:** {rouge_scores}")        
            st.write(f"**Perplexity Score:** {perplexity_scores}")
            st.write(f"**Bias Score:** {bias_score}")   
