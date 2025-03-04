# utils/analysis_utils.py
import streamlit as st
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from rouge import Rouge
from detoxify import Detoxify
from presidio_analyzer import AnalyzerEngine
from Dbias.bias_classification import classifier

rouge = Rouge()

def calculate_toxicity(response):
    return Detoxify("original").predict(response)

def calculate_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    return {
        'neutrality': sentiment_scores['neu'],
        'subjectivity': TextBlob(text).sentiment.subjectivity,
        'polarity': TextBlob(text).sentiment.polarity
    }

def rouge_score(response, context):
    scores = rouge.get_scores(response, context)
    return scores[0]["rouge-1"]["f"]

def bias_score_func(response):
    return classifier(response)

def detect_pii(response):
    engine = AnalyzerEngine()
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
    fig, ax = plt.subplots()
    ax.barh(toxicity_labels, toxicity_values, color=['green', 'yellow', 'orange', 'red', 'gray'])
    ax.set_xlabel("Toxicity Level")
    ax.set_title("Toxicity Analysis")
    st.pyplot(fig)

    max_toxicity = max(toxicity_values)
    if max_toxicity < 0.05:
        st.success("✅ Low Toxicity")
    elif max_toxicity < 0.1:
        st.warning("⚠️ Moderate Toxicity")
    else:
        st.error("🚨 High Toxicity")
