import streamlit as st

st.set_page_config(
    page_title="Create Post",
    page_icon="❓",
    layout="wide",
)
from utils.embedding_utils import (
    load_embedding_model, 
    add_documents_to_collection,
    query_chroma
)
from utils.web_page import fetch_data_from_urls
from utils.llm_utils import (
    detect_hallucination, generate_response, groundness_func, answer_relevance_func, 
    context_relevance_func, cosine_similarity_func, moderate_response, perplexity_score
)


from utils.analysis_utils import (
    Neutrality_viz, answer_relevance, calculate_toxicity, calculate_sentiment, context_relevance, maxx_toxicity, meteor_score_func, meteor_viz, perplexity_viz, pii_viz, polarity_viz, rouge_score, 
    bias_score_func, detect_pii, rouge_viz, subjectivity_viz, visualize_groundness, visualize_toxicity
)


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
            relevant_docs = query_chroma(question)
            context = "\n".join(relevant_docs) if relevant_docs else "No relevant context found."
            response = generate_response(question, context)
            # response = "this is a test response"
            
            # write the context and response in log file
            with open("log_query.txt", "a") as f:
                f.write(f"Context: {context}\n")
                f.write(f"Question: {question}\n")
                f.write(f"Response: {response}\n\n")
            
            
            # st.subheader("Generated Response")
            # st.write(response)
            # 📌 Keep Response Always Visible
            st.header("Generated Response")
            st.success(response)  # ✅ Keeps response on top in a highlighted box
            
           
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
            meteor_score = meteor_score_func(context,response)
            max_tox = maxx_toxicity(toxicity_score)
            moderated_response = None
            if(max_tox>0.1):
                   moderated_response = moderate_response(question,response)
            
            if moderated_response:
                toxicity_score2 = calculate_toxicity(moderated_response)
                sentiment_scores2 = calculate_sentiment(moderated_response)
                groundness_score2 = groundness_func(context, moderated_response)
                answer_relevance_score2 = answer_relevance_func(moderated_response, question)
                context_relevance_score2 = context_relevance_func(context, question)
                rouge_scores2 = rouge_score(moderated_response, context)
                perplexity_scores2 = perplexity_score(moderated_response)
                bias_score2 = bias_score_func(moderated_response)
                cosine_similarity_score2 = cosine_similarity_func(question, moderated_response)
                pii2 = detect_pii(moderated_response)
                meteor_score2 = meteor_score_func(context,moderated_response)
            
            
            

            if moderated_response:
                st.subheader("Moderated Response")
                st.success(moderated_response)
            # st.header("Responsible AI Analysis")
            # visualize_toxicity(toxicity_score)
            # st.write(f"**Toxicity Score:** {toxicity_score}")
            # work in progress ----------
            # if moderated_response:
            #     st.subheader("Moderated Response")
            #     st.write(moderated_response)

            # detect_hallucination(question, response)
            # st.write(f"**Cosine Similarity Score:** {cosine_similarity_score}")  
            # pii_viz(pii)
            # st.write(f"**PII Detection:** {pii}")
            # visualize_groundness(groundness_score)      
            # st.write(f"**Groundness Score:** {groundness_score}")
            # answer_relevance(answer_relevance_score)
            # st.write(f"**Answer Relevance Score:** {answer_relevance_score}")
            # # context_relevance(context_relevance_score)
            # # st.write(f"**Context Relevance Score:** {context_relevance_score}")
            # Neutrality_viz(sentiment_scores['neutrality'])
            # st.write(f"**Neutrality:** {sentiment_scores['neutrality']}")
            # subjectivity_viz(sentiment_scores['subjectivity'])
            # st.write(f"**Subjectivity:** {sentiment_scores['subjectivity']}")
            # polarity_viz(sentiment_scores['polarity'])            
            # st.write(f"**Polarity:** {sentiment_scores['polarity']}")            
            # rouge_viz(rouge_scores)
            # st.write(f"**ROUGE Score:** {rouge_scores}")               
            # # st.write(f"**Bias Score:** {bias_score}") 
            # meteor_viz(meteor_score) 
            # st.write(f"**Meteor Score:** {meteor_score['meteor']}")
            # perplexity_viz(perplexity_scores)
            # st.write(f"**Perplexity Score:** {perplexity_scores}")

            
            # 📌 Use Expander for Scrollable Analysis
            st.header("Responsible AI Analysis")

            with st.expander("📊 View Analysis Results", expanded=False):
                # --- Analysis Results ---
                visualize_toxicity(toxicity_score)
                # st.write(f"**Toxicity Score:** {toxicity_score}")

                detect_hallucination(question, response)
                st.write(f"**Cosine Similarity Score:** {cosine_similarity_score}")

                pii_viz(pii)
                st.write(f"**PII Detection:** {pii}")

                visualize_groundness(groundness_score)
                st.write(f"**Groundness Score:** {groundness_score}")

                answer_relevance(answer_relevance_score)
                st.write(f"**Answer Relevance Score:** {answer_relevance_score}")

                Neutrality_viz(sentiment_scores['neutrality'])
                st.write(f"**Neutrality:** {sentiment_scores['neutrality']}")

                subjectivity_viz(sentiment_scores['subjectivity'])
                st.write(f"**Subjectivity:** {sentiment_scores['subjectivity']}")

                polarity_viz(sentiment_scores['polarity'])
                st.write(f"**Polarity:** {sentiment_scores['polarity']}")                

                rouge_viz(rouge_scores)
                st.write(f"**ROUGE Score:** {rouge_scores}")
                
                meteor_viz(meteor_score)
                st.write(f"**Meteor Score:** {meteor_score}")

                perplexity_viz(perplexity_scores)
                st.write(f"**Perplexity Score:** {perplexity_scores}")

            if moderated_response:
                    # 📌 Use Expander for Scrollable Analysis
                    st.header("Responsible AI Analysis for a Moderated Response")

                    with st.expander("📊 View Analysis Results", expanded=False):
                        # --- Analysis Results ---
                        visualize_toxicity(toxicity_score2)
                        # st.write(f"**Toxicity Score:** {toxicity_score}")

                        detect_hallucination(question, moderated_response)
                        st.write(f"**Cosine Similarity Score:** {cosine_similarity_score2}")

                        pii_viz(pii)
                        st.write(f"**PII Detection:** {pii2}")

                        visualize_groundness(groundness_score2)
                        st.write(f"**Groundness Score:** {groundness_score2}")

                        answer_relevance(answer_relevance_score2)
                        st.write(f"**Answer Relevance Score:** {answer_relevance_score2}")

                        Neutrality_viz(sentiment_scores2['neutrality'])
                        st.write(f"**Neutrality:** {sentiment_scores2['neutrality']}")

                        subjectivity_viz(sentiment_scores2['subjectivity'])
                        st.write(f"**Subjectivity:** {sentiment_scores2['subjectivity']}")

                        polarity_viz(sentiment_scores2['polarity'])
                        st.write(f"**Polarity:** {sentiment_scores2['polarity']}")                

                        rouge_viz(rouge_scores2)
                        st.write(f"**ROUGE Score:** {rouge_scores2}")
                        
                        meteor_viz(meteor_score2)
                        st.write(f"**Meteor Score:** {meteor_score2}")

                        perplexity_viz(perplexity_scores2)
                        st.write(f"**Perplexity Score:** {perplexity_scores2}")
            
