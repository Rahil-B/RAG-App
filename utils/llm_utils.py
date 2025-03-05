# utils/llm_utils.py
import streamlit as st
import ollama
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import GROQ_API_KEY
import streamlit as st

@st.cache_resource
def get_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")

@st.cache_resource
def get_model():
    return GPT2LMHeadModel.from_pretrained("gpt2")

@st.cache_resource
def get_model_embed():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Initialize GPT2 resources (for perplexity)
tokenizer = get_tokenizer()#GPT2Tokenizer.from_pretrained("gpt2")
model = get_model()#GPT2LMHeadModel.from_pretrained("gpt2")
model_embed = get_model_embed()#SentenceTransformer('all-MiniLM-L6-v2')
# Initialize ChatGroq LLM
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

# def generate_response(question, context):
#     prompt = f"Context: {context}\nQuestion: {question}\nAnswer strictly based on the given context."
#     response = ollama.chat(model="llama2-uncensored", messages=[{"role": "user", "content": prompt}])
#     return response["message"]["content"]

def generate_response(question, context):
    # prompt = f"Context: {context}\nQuestion: {question}\nAnswer strictly based on the given context."
    prompt = [
    {"role": "system", "content": """A user will give you a context, and you must generate a high-quality, engaging social media post strictly based on that context.
    
    - **Do not add any information that is not present in the context.**
    - Ensure that the post is relevant, engaging, and formatted for online posting.
    - Add **4-6 relevant hashtags** to increase engagement.
    - Use a **conversational and catchy tone** while maintaining factual accuracy.
    - If the context is missing or lacks enough information, say **'Sorry, no relevant context provided'** instead of generating a post."""},    
    {"role": "user", "content": "Context: " + str(context) + " question: " + str(question)}
    ]
    # response = ollama.chat(model="mistral", messages=prompt)
    response = ollama.chat(model="mistral", messages=prompt)
    return response["message"]["content"]

def generate_moderated_response(question, context):
    # prompt = f"Context: {context}\nQuestion: {question}\nAnswer strictly based on the given context."
    prompt = [
    {"role": "system", "content": """A user will give you a context, and you must generate a high-quality, engaging social media post strictly based on that context.
    
    - **Do not add any information that is not present in the context.**
    - Ensure that the post is relevant, engaging, and formatted for online posting.
    - Add **4-6 relevant hashtags** to increase engagement.
    - Use a **conversational and catchy tone** while maintaining factual accuracy.
    - If the context is missing or lacks enough information, say **'Sorry, no relevant context provided'** instead of generating a post."""},    
    {"role": "user", "content": "Context: " + str(context) + " question: " + str(question)}
    ]
    # response = ollama.chat(model="mistral", messages=prompt)
    response = ollama.chat(model="mistral", messages=prompt)
    return response["message"]["content"]


def groundness_func(context, response):
    prompt = [
        SystemMessage(content="""A user will give you a context and a response.
Your task is to rate how much the response is supported by the context.
Your answer must be **only** a number between 0.00 and 1.00 rounded to two decimal places. 
0.00 = Response is **not at all** supported by the context.
1.00 = Response is **fully** supported by the context.
**DO NOT** include any additional text."""),
        HumanMessage(content=f"Context: {context}\nResponse: {response}")
    ]
    groundness = llm.invoke(prompt)
    return groundness.content

def answer_relevance_func(response, question):
    prompt = [
        {"role": "system", "content": """A user will give you a response and a question \
and your task is to rate how relevant the response is to the question. Your answer must be only \
a number between 0.0 and 1.0 rounded to two decimal places, with no additional text."""}, 
        {"role": "user", "content": f"Response: {response} Question: {question}"}
    ]
    return llm.invoke(prompt).content

def context_relevance_func(context, question):
    prompt = [
        {"role": "system", "content": """A user will give you a context and a question \
and your task is to rate how relevant the context is to the question. Your answer must be only \
a number between 0.0 and 1.0 rounded to two decimal places, with no additional text."""}, 
        {"role": "user", "content": f"Context: {context} Question: {question}"}
    ]
    return llm.invoke(prompt).content

def cosine_similarity_func(question, response):
    
    query_embedding = model_embed.encode([question])
    response_embedding = model_embed.encode([response])
    cosine_sim = cosine_similarity(query_embedding, response_embedding)
    return cosine_sim[0][0]

def detect_hallucination(question, response):
    st.subheader("hallucinations detection")
    cosine_sim_score = cosine_similarity_func(question, response)
    if cosine_sim_score > 0.4:
        st.success("✅ No Hallucination")
        return "No Hallucination"
    else:
        st.error("🚨 Hallucinated")
        return "Hallucinated"

def perplexity_score(text):
    inputs = tokenizer(text, return_tensors="pt")
    loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()
