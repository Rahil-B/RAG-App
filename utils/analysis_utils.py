# utils/analysis_utils.py
import streamlit as st
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from rouge import Rouge
from detoxify import Detoxify
from presidio_analyzer import AnalyzerEngine
from Dbias.bias_classification import classifier
import requests
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def get_rouge():
    return Rouge()
#rouge = Rouge()
@st.cache_resource
def get_sia():
    return SentimentIntensityAnalyzer()
#sia = SentimentIntensityAnalyzer()
@st.cache_resource
def get_engine():
    return AnalyzerEngine()



# rouge = Rouge()


API_KEY = "AIzaSyAaiBWopGwFvYW4Poc-MdjZMz5bgbHQzCQ"  # Replace with your actual API key

# Google Perspective API endpoint
url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={API_KEY}"

def calculate_toxicity(text):
    """Moderates text using Google Perspective API."""
    data = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {
            "TOXICITY": {},  # Main moderation category
            "SEVERE_TOXICITY": {},
            "INSULT": {},
            "THREAT": {},
            "IDENTITY_ATTACK": {},
            "SEXUALLY_EXPLICIT": {}
        }
    }

    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        scores = response.json().get("attributeScores", {})
        return {attr: scores[attr]["summaryScore"]["value"] for attr in scores}
    else:
        return {"error": f"API Error: {response.status_code}"}

def calculate_sentiment(text):
    sia=get_sia()
    sentiment_scores = sia.polarity_scores(text)
    return {
        'neutrality': sentiment_scores['neu'],
        'subjectivity': TextBlob(text).sentiment.subjectivity,
        'polarity': TextBlob(text).sentiment.polarity
    }

def rouge_score(response, context):
    rouge=get_rouge()
    scores = rouge.get_scores(response, context)
    return scores[0]["rouge-1"]["f"]

def bias_score_func(response):
    return classifier(response)

def detect_pii(response):
    engine=get_engine()
    results = engine.analyze(
        text=response,
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN", "URL", "LOCATION", "IP_ADDRESS"],
        language="en"
    )
    detected_pii = []
    for result in results:
        pii_value = response[result.start:result.end]
        detected_pii.append(f"Type: {result.entity_type}, Value: {pii_value}, Score: {result.score:.2f}")
    return detected_pii

def visualize_toxicity(toxicity_score):
    toxicity_labels = list(toxicity_score.keys())
    toxicity_values = list(toxicity_score.values())

    st.subheader("Toxicity Score Visualization")
    # fig, ax = plt.subplots()
    # ax.barh(toxicity_labels, toxicity_values, color=['green', 'yellow', 'orange', 'red', 'gray'])
    # ax.set_xlabel("Toxicity Level")
    # ax.set_title("Toxicity Analysis")
    # st.pyplot(fig)

    max_toxicity = max(toxicity_values)
    if max_toxicity < 0.05:
        st.success("✅ Low Toxicity")
    elif max_toxicity < 0.1:
        st.warning("⚠️ Moderate Toxicity")
    else:
        st.error("🚨 High Toxicity")


def visualize_groundness(groundness_score):
    st.subheader("groundness detection")
    if float(groundness_score) > 0.7:
        st.success("✅ Grounded")
    elif float(groundness_score)  == 0.0:
        st.warning("⚠️ No Context was provided for this response")
    else:
        st.error("🚨 Ungrounded")

def answer_relevance(answer_relevance_score):
    st.subheader("Answer Relevance Detection")
    if float(answer_relevance_score) > 0.7:
        st.success("✅ Answer is strongly Relevant")
    elif float(answer_relevance_score) > 0.5 and float(answer_relevance_score) < 0.7:
        st.warning("⚠️ Answer is Moderately Relevant")
    else:
        st.error("🚨 Answer is not Relevant")

def context_relevance(context_relevance_score):
    st.subheader("Context Relevence Detection")
    if float(context_relevance_score) > 0.7:
        st.success("✅ Answer is strongly Relevant to context")
    elif context_relevance_score > 0.5 and float(context_relevance_score) < 0.7:
        st.warning("⚠️ Answer is Moderately Relevant to context")
    elif float(context_relevance_score) == 0.0:
        st.warning("⚠️ No Context was provided for this response")
    else:
        st.error("🚨 Answer is not Relevant to context")

def Neutrality_viz(score):
    st.subheader("Answer Neutralty Detection")
    if float(score) > 0.7:
        st.success("✅ Answer is Neutral")
    elif float(score) > 0.5 and float(score) < 0.7:
        st.warning("⚠️ Answer is Moderately Neutral")
    else:
        st.error("🚨 Answer is not Neutral")

def answer_relevance(answer_relevance_score):
    st.subheader("Answer Relevance Detection")
    if float(answer_relevance_score) > 0.7:
        st.success("✅ Answer is strongly Relevant")
    elif float(answer_relevance_score) > 0.5 and float(answer_relevance_score) < 0.7:
        st.warning("⚠️ Answer is Moderately Relevant")
    else:
        st.error("🚨 Answer is not Relevant")

def subjectivity_viz(score):
    st.subheader("Subjectivity Detection")
    if float(score) > 0.7:
        st.success("✅ Answer is strongly Relevant")
    elif float(score) > 0.5 and float(score) < 0.7:
        st.warning("⚠️ Answer is Moderately Relevant")
    else:
        st.error("🚨 Answer is not Relevant")
