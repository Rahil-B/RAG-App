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
import evaluate 
import pandas as pd

# 🔹 Tooltip Descriptions
groundness_tooltip_text = "Indicates whether the response is based on provided context. Higher scores mean more grounded responses."
answer_relevance_tooltip_text = "Measures how well the response answers the given question. Higher scores indicate better relevance."
context_relevance_tooltip_text = "Evaluates how relevant the context is to the response. A score of 0 means no context was provided."
neutrality_tooltip_text = "Determines whether the response maintains a neutral tone. Higher scores indicate less bias."
subjectivity_tooltip_text = "Indicates whether the response contains personal opinions or is objective."
polarity_tooltip_text = "Analyzes sentiment polarity. Positive values indicate optimism, negative values indicate criticism."
pii_tooltip_text = "Detects Personal Identifiable Information (PII) in the response."
rouge_tooltip_text = "Measures the overlap between the response and the context. Higher scores indicate better alignment."
meteor_tooltip_text = "Evaluates the relevance of the response to the reference text. Higher scores indicate better alignment."
perplexity_tooltip_text = "Measures how well the response is predicted by the language model. Lower scores indicate more natural text."
toxicity_tooltip_text = "Analyzes the toxicity of the response, measuring offensive language, threats, and hate speech."
polarity_tooltip_text = "Analyzes the sentiment of the response. Positive values indicate optimism, negative values indicate criticism."



# 🔹 Inject Global CSS (Once)
st.markdown(
    """
    <style>
    /* Tooltip Styling */
    .tooltip {
        position: relative;
        display: inline-flex;  /* Align inline with text */
        cursor: pointer;
        margin-left: 10px;  /* Add spacing between heading and ℹ️ */
        font-size: 18px;  /* Make the ℹ️ icon larger */
        font-weight: bold;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 230px;
        background-color: #1e1e1e;
        color: #fff;
        text-align: center;
        padding: 6px;
        border-radius: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -115px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 13px;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 🔹 Tooltip Function (Now Properly Spaced & Bigger)
def add_tooltip(text):
    """Returns an inline tooltip wrapped around an enlarged ℹ️ icon."""
    return f"""
    <span class="tooltip"> ℹ️
        <span class="tooltiptext">{text}</span>
    </span>
    """

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

    # st.subheader("Toxicity Score Visualization")

    st.markdown(
        f"""<h3 style="display: inline-flex; align-items: center;">Toxicity Score Visualization {add_tooltip(toxicity_tooltip_text)}</h3>""",
        unsafe_allow_html=True
    )

    # fig, ax = plt.subplots()
    # ax.barh(toxicity_labels, toxicity_values, color=['green', 'yellow', 'orange', 'red', 'gray'])
    # ax.set_xlabel("Toxicity Level")
    # ax.set_title("Toxicity Analysis")
    # st.pyplot(fig)

    
    # Toxicity Level Analysis
    max_toxicity = max(toxicity_score.values())

    # Create DataFrame (2 Rows)
    df = pd.DataFrame([toxicity_values], columns=toxicity_labels)
    df.index = [""]
    # Display Table
    st.table(df) 

    # max_toxicity = max(toxicity_values)
    if max_toxicity < 0.1:
        st.success("✅ Low Toxicity")
    elif max_toxicity < 0.5:
        st.warning("⚠️ Moderate Toxicity")
    else:
        st.error("🚨 High Toxicity")
    
        
def maxx_toxicity(toxicity_score):
    toxicity_labels = list(toxicity_score.keys())
    toxicity_values = list(toxicity_score.values())
    max_toxicity = max(toxicity_values)
    return max_toxicity

def meteor_score_func(context,response):
  meteor = evaluate.load('meteor')
  meteor_score = meteor.compute(predictions=[response], references=[context])
  return meteor_score


def visualize_groundness(groundness_score):
    # st.subheader("groundness detection")
    st.markdown(
        f"""<h3 style="display: inline-flex; align-items: center;">Groundness Detection {add_tooltip(groundness_tooltip_text)}</h3>""",
        unsafe_allow_html=True
    )

    if float(groundness_score) > 0.7:
        st.success("✅ Grounded")
    elif float(groundness_score) > 0.3 and float(groundness_score) < 0.7:
        st.warning("⚠️ Moderately Grounded")
    elif float(groundness_score)  == 0.0:
        st.warning("⚠️ No Context was provided for this response")
    else:
        st.error("🚨 Ungrounded")

def answer_relevance(answer_relevance_score):
    # st.subheader("Answer Relevance Detection")

    st.markdown(
        f"""<h3 style="display: inline-flex; align-items: center;">Answer Relevance Detection {add_tooltip(answer_relevance_tooltip_text)}</h3>""",
        unsafe_allow_html=True
    )

    if float(answer_relevance_score) > 0.7:
        st.success("✅ Answer is strongly Relevant")
    elif float(answer_relevance_score) > 0.5 and float(answer_relevance_score) < 0.7:
        st.warning("⚠️ Answer is Moderately Relevant")
    else:
        st.error("🚨 Answer is not Relevant")

def context_relevance(context_relevance_score):
    # st.subheader("Context Relevence Detection")
    st.markdown(
        f"""<h3 style="display: inline-flex; align-items: center;">Context Relevance Detection {add_tooltip(context_relevance_tooltip_text)}</h3>""",
        unsafe_allow_html=True
    )

    if float(context_relevance_score) > 0.7:
        st.success("✅ Answer is strongly Relevant to context")
    elif float(context_relevance_score) > 0.5 and float(context_relevance_score) < 0.7:
        st.warning("⚠️ Answer is Moderately Relevant to context")
    elif float(context_relevance_score) == 0.0:
        st.warning("⚠️ No Context was provided for this response")
    else:
        st.error("🚨 Answer is not Relevant to context")

def Neutrality_viz(score):
    # st.subheader("Answer Neutralty Detection")
    st.markdown(
        f"""<h3 style="display: inline-flex; align-items: center;">Answer Neutrality Detection {add_tooltip(neutrality_tooltip_text)}</h3>""",
        unsafe_allow_html=True
    )

    if float(score) > 0.7:
        st.success("✅ Answer is Neutral")
    elif float(score) > 0.5 and float(score) < 0.7:
        st.warning("⚠️ Answer is Moderately Neutral")
    else:
        st.error("🚨 Answer is not Neutral")

def answer_relevance(answer_relevance_score):
    # st.subheader("Answer Relevance Detection")
    st.markdown(
        f"""<h3 style="display: inline-flex; align-items: center;">Answer Relevance Detection {add_tooltip(answer_relevance_tooltip_text)}</h3>""",
        unsafe_allow_html=True
    )

    if float(answer_relevance_score) > 0.7:
        st.success("✅ Answer is strongly Relevant")
    elif float(answer_relevance_score) > 0.5 and float(answer_relevance_score) < 0.7:
        st.warning("⚠️ Answer is Moderately Relevant")
    else:
        st.error(f"🚨 Answer is not Relevant")

def subjectivity_viz(score):
    # st.subheader("Subjectivity Detection")
    st.markdown(
        f"""<h3 style="display: inline-flex; align-items: center;">Subjectivity Detection {add_tooltip(subjectivity_tooltip_text)}</h3>""",
        unsafe_allow_html=True
    )

    score = float(score)
    
    if score > 0.7:
        st.error(f"🚨 Highly Subjective: The answer contains strong personal opinions or emotions")
    elif 0.5 < score <= 0.7:
        st.warning(f"⚠️ Moderately Subjective: The answer has a mix of opinions and objective statements")
    else:
        st.success(f"✅ Low Subjectivity: The answer is mostly objective and factual")

def polarity_viz(score):
    # st.subheader("Polarity Detection")
    st.markdown(
        f"""<h3 style="display: inline-flex; align-items: center;">Polarity Detection {add_tooltip(polarity_tooltip_text)}</h3>""",
        unsafe_allow_html=True
    )
    score = float(score)
    
    if score > 0.5:
        st.success(f"✅ Positive Sentiment: The answer is positive and optimistic")
    elif -0.5 <= score <= 0.5:
        st.success(f"⚠️ Neutral Sentiment: The answer is balanced or mixed")
    else:
        st.warning(f"🚨 Negative Sentiment: The answer has a negative or critical tone")

def pii_viz(pii):
    # st.subheader("PII Detection")
    st.markdown(
        f"""<h3 style="display: inline-flex; align-items: center;">PII Detection {add_tooltip(pii_tooltip_text)}</h3>""",
        unsafe_allow_html=True
    )

    if pii:
        st.error("🚨 Personal Identifiable Information Detected:")
        for entity in pii:
            st.write(entity)
    else:
        st.success("✅ No PII Detected")

def rouge_viz(score):
    # st.subheader("ROUGE Score Analysis")
    st.markdown(
        f"""<h3 style="display: inline-flex; align-items: center;">ROUGE Score Analysis {add_tooltip(rouge_tooltip_text)}</h3>""",
        unsafe_allow_html=True
    )
    
    score = float(score)
    
    if score > 0.5:
        st.success(f"✅ High Overlap: The response is highly aligned with the context")
    elif 0.3 < score <= 0.5:
        st.warning(f"⚠️ Moderate Overlap: The response partially aligns with the context")
    else:
        st.error(f"🚨 Low Overlap: The response has minimal alignment with the context")


def meteor_viz(score):
    # st.subheader("Meteor Score based Analysis")
    st.markdown(
        f"""<h3 style="display: inline-flex; align-items: center;">Meteor Score Analysis {add_tooltip(meteor_tooltip_text)}</h3>""",
        unsafe_allow_html=True
    )
    score = float(score["meteor"])
    
    if score > 0.5:
        st.success(f"✅ The response is highly relevant and well-aligned with the reference.")
    elif 0.2 < score <= 0.5:
        st.warning(f"⚠️ The response is somewhat relevant but could be improved.")
    else:
        st.error(f"🚨 The response is poorly aligned with the reference.")

def perplexity_viz(score):
    # st.subheader("Perplexity Score Analysis")
    st.markdown(
        f"""<h3 style="display: inline-flex; align-items: center;">Perplexity Score Analysis {add_tooltip(perplexity_tooltip_text)}</h3>""",
        unsafe_allow_html=True
    )
    
    score = float(score)
    
    if score < 30:
        st.success(f"✅ Low Perplexity: The response is highly natural and well-predicted by the model")
    elif 30 <= score <= 60:
        st.warning(f"⚠️ Moderate Perplexity: The response is somewhat predictable but could be improved")
    else:
        st.error(f"🚨 High Perplexity: The response is difficult for the model to predict, indicating unnatural or confusing text")