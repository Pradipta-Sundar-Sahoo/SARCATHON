from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
import openai
import json
from mangum import Mangum
import os

from dotenv import load_dotenv
load_dotenv()


OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY']=OPENAI_API_KEY


# Initialize FastAPI app
app = FastAPI()
magnum_app = Mangum(app)

# Load the index and data
index = faiss.read_index('faq_index.faiss')
questions = np.load('questions.npy', allow_pickle=True)
answers = np.load('answers.npy', allow_pickle=True)

# Load API key
openai.api_key = OPENAI_API_KEY

# Load FAQs
with open('faqs.json') as f:
    faqs = json.load(f)

faq_data = {}
for category, entries in faqs.items():
    faq_data[category] = {
        'questions': [faq['question'] for faq in entries],
        'answers': [faq['answer'] for faq in entries]
    }

# Pydantic model for input queries
class Query(BaseModel):
    user_query: str

def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return np.array(response['data'][0]['embedding']).astype('float32')

@app.post("/concise_answer/")
async def fetch_concise_answer(query: Query):
    if not query.user_query:
        raise HTTPException(status_code=400, detail="Please enter a question.")
    
    rewritten_query = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": (
                "Please rewrite the following query to enhance clarity and conciseness while maintaining the original intent. "
                "Ensure that any complex phrases are simplified:\n"
                f"{query.user_query}"
            )
        }]
    )['choices'][0]['message']['content']

    query_embedding = get_embedding(rewritten_query).reshape(1, -1)
    D, I = index.search(query_embedding, k=5) 
    fetched_answers = answers[I[0]]

    reranked_answers = []
    for answer in fetched_answers:
        answer_embedding = get_embedding(answer).reshape(1, -1)
        score = np.dot(query_embedding, answer_embedding.T) 
        reranked_answers.append((score, answer))

    reranked_answers.sort(key=lambda x: x[0], reverse=True)
    concise_answer = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": (
                "Please provide a concise, structured answer in a maximum of three lines. "
                "Ensure the response is clear and focuses on the relevant details. "
                "Check for any numerical data and present it accurately based on the following information:\n"
                f"{reranked_answers}"
            )
        }])['choices'][0]['message']['content']
    
    return {"concise_answer": concise_answer}

@app.post("/detailed_answer/")
async def fetch_detailed_answer(query: Query):
    if not query.user_query:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    rewritten_query = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": (
                "Please rewrite the following query to enhance clarity and conciseness while maintaining the original intent. "
                "Ensure that any complex phrases are simplified:\n"
                f"{query.user_query}"
            )
        }]
    )['choices'][0]['message']['content']

    query_embedding = get_embedding(rewritten_query).reshape(1, -1)
    D, I = index.search(query_embedding, k=5) 
    fetched_answers = answers[I[0]]

    reranked_answers = []
    for answer in fetched_answers:
        answer_embedding = get_embedding(answer).reshape(1, -1)
        score = np.dot(query_embedding, answer_embedding.T) 
        reranked_answers.append((score, answer))

    reranked_answers.sort(key=lambda x: x[0], reverse=True)
    top_answers = [answer for _, answer in reranked_answers]

    detailed_answer = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": (
                "Please provide a detailed and well-structured answer in English. "
                "Ensure that all numerical data is accurate and clearly presented. "
                "Based on the following information, elaborate on the key points and insights:\n"
                f"{reranked_answers}"
            )
        }])['choices'][0]['message']['content']

    return {"detailed_answer": detailed_answer}

@app.post("/similar_questions/")
async def fetch_similar_questions(query: Query):
    if not query.user_query:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    rewritten_query = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": (
                "Please rewrite the following query to enhance clarity and conciseness while maintaining the original intent. "
                "Ensure that any complex phrases are simplified:\n"
                f"{query.user_query}"
            )
        }]
    )['choices'][0]['message']['content']

    query_embedding = get_embedding(rewritten_query).reshape(1, -1)
    D, I = index.search(query_embedding, k=5) 
    similar_questions = questions[I[0]]

    results = []
    for simi in similar_questions:
        for category, entries in faqs.items():
            for i, question in enumerate(entries):
                if simi == question['question']:  
                    results.append({
                        "question": simi,
                        "answer": question['answer']
                    })
                    break  

    return {"similar_questions": results}

# Run the app using: uvicorn filename:app --reload
