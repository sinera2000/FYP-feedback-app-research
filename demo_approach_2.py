# demo_approach_2.py
# A simplified, one-question app to demonstrate the LLM Keyword Summary method.

import os
import random
import re
import joblib
import numpy as np
import shap
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
from groq import Groq
import nltk
from nltk.corpus import stopwords
from bson import ObjectId # Import ObjectId to check for it
import json

# --- Custom JSON Encoder to handle ObjectId ---
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

# --- 1. INITIAL SETUP & MODEL LOADING ---
print("--- ⏳ Initializing Demo for Approach 2 ---")
load_dotenv()
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.json_encoder = JSONEncoder # Use the custom encoder

# --- Connect to Services & Load Models ---
MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
db = MongoClient(MONGO_URI)["InterviewBotDB"]
questions_collection = db["questions"]
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client_groq = Groq(api_key=GROQ_API_KEY)
LLM_MODEL = "llama-3.1-8b-instant"
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
classifier = CatBoostClassifier()
classifier.load_model("catboost_classifier.cbm")
label_encoder = joblib.load('label_encoder.pkl')
print("✅ Models, DB, and Groq client ready.")

# --- 2. CORE EVALUATION & FEEDBACK FUNCTIONS ---
def preprocess_text(text):
    if not isinstance(text, str): text = str(text)
    return re.sub(r'[^\w\s]', '', text.lower())

def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def run_evaluation(user_answer, ideal_answer):
    processed_user = preprocess_text(user_answer)
    processed_ideal = preprocess_text(ideal_answer)
    user_embedding = sbert_model.encode(processed_user)
    ideal_embedding = sbert_model.encode(processed_ideal)
    sim_score = util.cos_sim(user_embedding, ideal_embedding).item()
    jaccard = jaccard_similarity(processed_user, processed_ideal)
    word_diff = len(processed_user.split()) - len(processed_ideal.split())
    features = [[sim_score, jaccard, word_diff]]
    prediction_index = classifier.predict(features)[0][0]
    evaluation_result = label_encoder.inverse_transform([prediction_index])[0]
    return {"evaluation": evaluation_result}

def get_positive_shap_explanation(user_answer, ideal_answer):
    def predictor(texts):
        processed_texts = [preprocess_text(t) for t in texts]
        processed_ideal = preprocess_text(ideal_answer)
        text_embeddings = sbert_model.encode(processed_texts)
        ideal_embedding = sbert_model.encode(processed_ideal)
        features = []
        for i in range(len(processed_texts)):
            sim_score = util.cos_sim(text_embeddings[i], ideal_embedding).item()
            jaccard = jaccard_similarity(processed_texts[i], processed_ideal)
            word_diff = len(processed_texts[i].split()) - len(processed_ideal.split())
            features.append([sim_score, jaccard, word_diff])
        return classifier.predict_proba(np.array(features))
    explainer = shap.Explainer(predictor, shap.maskers.Text())
    shap_values = explainer([user_answer])
    predicted_class_index = np.argmax(shap_values.values[0].sum(axis=0))
    positive_words = [word for i, word in enumerate(shap_values.data[0]) if shap_values.values[0][i][predicted_class_index] > 0]
    return [word for word in positive_words if word.lower() not in stop_words and len(word) > 2]

def get_missing_keywords(user_answer, ideal_answer):
    user_answer_str = str(user_answer)
    ideal_answer_str = str(ideal_answer)
    ideal_keywords = set(re.findall(r'\w+', ideal_answer_str.lower())) - stop_words
    user_keywords = set(re.findall(r'\w+', user_answer_str.lower())) - stop_words
    missing = ideal_keywords - user_keywords
    return sorted([word for word in missing if isinstance(word, str) and len(word) > 2])

# --- APPROACH 2 FEEDBACK LOGIC ---
def summarize_strengths_with_groq(user_answer, positive_keywords):
    if not positive_keywords: return None
    positive_str = ", ".join(positive_keywords)
    prompt = f"""
    Based on the user's answer and a list of positive keywords identified by an AI, write a single, encouraging, human-readable sentence summarizing what the user did well.
    User Answer: "{user_answer}"
    Positive Keywords: [{positive_str}]
    Example Output: "Your answer was strong when you correctly mentioned concepts like reusability and type safety."
    """
    try:
        chat_completion = client_groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}], model=LLM_MODEL
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq strengths summary failed: {e}")
        return f"Your answer was strong when you mentioned: {', '.join(positive_keywords)}" # Fallback

def generate_feedback_approach2(evaluation, user_answer, ideal_answer):
    feedback = ""
    if evaluation == 'correct':
        return "✅ Excellent! Your answer was comprehensive and accurate."
    
    missing_keywords = get_missing_keywords(user_answer, ideal_answer)
    
    if evaluation == 'partially correct':
        feedback += "This is a good answer that is on the right track.\n"
        positive_keywords = get_positive_shap_explanation(user_answer, ideal_answer)
        strengths_summary = summarize_strengths_with_groq(user_answer, positive_keywords)
        if strengths_summary:
            feedback += f"\n✅ **What you did well:** {strengths_summary}\n"

    if missing_keywords:
         feedback += f"\n❌ **To improve:** Your answer could be more complete by including key ideas like: **{', '.join(missing_keywords[:5])}**."

    return feedback

# --- 3. FLASK WEB ROUTES FOR DEMO ---
@app.route('/')
def index():
    question_doc = list(questions_collection.aggregate([{"$sample": {"size": 1}}]))[0]
    
    # --- FIX: Convert ObjectId to string before saving to session ---
    if '_id' in question_doc:
        question_doc['_id'] = str(question_doc['_id'])
        
    session['question_doc'] = question_doc
    return render_template('demo_interview.html', question=question_doc['Question'])

@app.route('/submit', methods=['POST'])
def submit():
    user_answer = request.form['user_answer']
    question_doc = session['question_doc']
    
    evaluation = run_evaluation(user_answer, question_doc['Ideal Answer'])
    feedback = generate_feedback_approach2(evaluation['evaluation'], user_answer, question_doc['Ideal Answer'])
    
    return render_template('demo_feedback.html', 
                           question=question_doc['Question'],
                           user_answer=user_answer,
                           evaluation=evaluation['evaluation'],
                           feedback=feedback,
                           approach_name="Approach 2: LLM Keyword Summary")

if __name__ == '__main__':
    app.run(debug=True, port=5005)
