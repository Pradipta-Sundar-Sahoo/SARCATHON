import streamlit as st
import numpy as np
import faiss
import openai
import json
import os
from dotenv import load_dotenv
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_lottie import st_lottie
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Set up output directory for audio files
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load API key for OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# Load FAISS index and data files
index = faiss.read_index('faq_index.faiss')
questions = np.load('questions.npy', allow_pickle=True)
answers = np.load('answers.npy', allow_pickle=True)

# Load FAQ data from JSON
with open('faqs.json') as f:
    faqs = json.load(f)

# Preprocess FAQ data for easy access
faq_data = {}
for category, entries in faqs.items():
    faq_data[category] = {
        'questions': [faq['question'] for faq in entries],
        'answers': [faq['answer'] for faq in entries]
    }

# Function to generate embeddings for a given text
def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return np.array(response['data'][0]['embedding']).astype('float32')


# Generate concise answer using OpenAI API
def generate_concise_answer(reranked_answers):
    concise_answer = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": (
                "Provide a concise answer in 3 lines, focusing on relevant details:\n"
                f"{reranked_answers}"
            )
        }]
    )['choices'][0]['message']['content']
    return concise_answer

# Generate detailed answer using OpenAI API
def generate_detailed_answer(reranked_answers):
    detailed_answer = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": (
                "Provide a detailed and structured answer:\n"
                f"{reranked_answers}"
            )
        }]
    )['choices'][0]['message']['content']
    return detailed_answer

# Streamlit app setup
st.set_page_config(page_title="SARAS AI INSTITUTE- FAQ Query System", page_icon="🔍", layout="centered")
st.markdown("<style>body {background-color: #f4f4f9;}</style>", unsafe_allow_html=True)

# Load animation for Streamlit app
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
lottie_animation = load_lottie_file("Animation - 1729046578679.json")

# Display header and input section
col1, col2 = st.columns([2, 1])
with col1:
    colored_header("SARAS AI INSTITUTE- FAQs", color_name="blue-70", description=None)
    st.markdown("Type your question below to get concise and detailed answers based on our FAQ database.")
with col2:
    st_lottie(lottie_animation, height=200)

# Get user input query
user_query = st.text_input("Enter your question:", placeholder="Ask me anything...")
add_vertical_space(2)

# If user submitted a query
if user_query and st.button("Submit"):
    with st.spinner("Processing your query..."):
        # Rewrite the query for better clarity
        rewritten_query = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": (
                    "Rewrite the following query to enhance clarity and conciseness:\n"
                    f"{user_query}"
                )
            }]
        )['choices'][0]['message']['content']

        # Generate embedding for the rewritten query
        query_embedding = get_embedding(rewritten_query).reshape(1, -1)

        # Search for similar questions in FAISS index
        D, I = index.search(query_embedding, k=5)
        similar_questions = questions[I[0]]
        fetched_answers = answers[I[0]]

        # Rerank the answers based on similarity score
        reranked_answers = []
        for answer in fetched_answers:
            answer_embedding = get_embedding(answer).reshape(1, -1)
            score = np.dot(query_embedding, answer_embedding.T)
            reranked_answers.append((score, answer))
        reranked_answers.sort(key=lambda x: x[0], reverse=True)
        top_answers = [answer for _, answer in reranked_answers]

        # Generate concise and detailed answers concurrently
        with ThreadPoolExecutor() as executor:
            future_concise = executor.submit(generate_concise_answer, reranked_answers)
            future_detailed = executor.submit(generate_detailed_answer, reranked_answers)
            concise_answer = future_concise.result()
            detailed_answer = future_detailed.result()

    # Display answers and similar questions
    st.markdown("---")
    colored_header("Concise Answer", color_name="green-70", description=None)
    st.success(concise_answer)
    st.markdown("---")
    colored_header("Detailed Answer", color_name="blue-30", description=None)
    st.info(detailed_answer)
    st.markdown("---")
    colored_header("Similar Questions", color_name="violet-50", description=None)
    for simi in similar_questions:
        st.subheader(simi)
        for category, entries in faqs.items():
            for i, question in enumerate(entries):
                if simi == question['question']:
                    st.markdown(f"**Answer:** {question['answer']}")
                    break
    st.markdown("---")
    add_vertical_space(2)
else:
    st.info("Enter your query above and click 'Submit' to get answers.")

# Display footer image
st.image("5.jpg")
