import json
import openai
import numpy as np
import faiss

# Load your FAQ JSON data
with open('faqs.json') as f:
    faq_data = json.load(f)

# Initialize OpenAI API
openai.api_key = 'sk-WDSyCWCXl0xsKpmsEc0zMu8FNWbc48aT9O3PvyR9drT3BlbkFJ7QToBIgUFcvhGF7eZgcpn8qTCIfmBlu1yeNobILF4A'  # Replace with your OpenAI API key

# Function to create embeddings for FAQ questions
def get_embeddings(texts):
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings).astype('float32')

# Prepare questions and answers
questions = []
answers = []

for category, items in faq_data.items():
    for item in items:
        questions.append(item['question'])
        answers.append(item['answer'])

# Create embeddings for the questions
question_embeddings = get_embeddings(questions)

# Build the FAISS index
index = faiss.IndexFlatL2(question_embeddings.shape[1])  # L2 distance
index.add(question_embeddings)

# Save the index
faiss.write_index(index, 'faq_index.faiss')

# Save questions and answers for retrieval
with open('questions.npy', 'wb') as f:
    np.save(f, np.array(questions))
with open('answers.npy', 'wb') as f:
    np.save(f, np.array(answers))

print("FAQ index and data saved successfully.")
