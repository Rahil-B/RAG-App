# utils/llm_utils.py
import ollama
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import GROQ_API_KEY

# Initialize GPT2 resources (for perplexity)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Initialize ChatGroq LLM
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

def generate_response(question, context):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer strictly based on the given context."
    response = ollama.chat(model="llama2-uncensored", messages=[{"role": "user", "content": prompt}])
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
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([question])
    response_embedding = model.encode([response])
    cosine_sim = cosine_similarity(query_embedding, response_embedding)
    return cosine_sim[0][0]

def perplexity_score(text):
    inputs = tokenizer(text, return_tensors="pt")
    loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()
