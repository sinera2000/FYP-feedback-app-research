# demo_approach_1.py
# A simplified, one-question app to demonstrate the Rule-Based Counterfactual method.

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
print("--- ⏳ Initializing Demo for Approach 1 ---")
load_dotenv()
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.json_encoder = JSONEncoder # Use the custom encoder

# --- Connect to Services & Load Models ---
MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
db = MongoClient(MONGO_URI)["InterviewBotDB"]
questions_collection = db["questions"]
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
classifier = CatBoostClassifier()
classifier.load_model("catboost_classifier.cbm")
label_encoder = joblib.load('label_encoder.pkl')
print("✅ Models and DB connection ready.")

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

def get_negative_shap_explanation(user_answer, ideal_answer):
    def predictor(texts):
        # ... (full predictor logic as in your main app) ...
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
    try:
        incorrect_class_index = list(label_encoder.classes_).index('incorrect')
        negative_words = [word for i, word in enumerate(shap_values.data[0]) if shap_values.values[0][i][incorrect_class_index] > 0]
        return [word for word in negative_words if word.lower() not in stop_words and len(word) > 2]
    except ValueError:
        return []

def get_missing_keywords(user_answer, ideal_answer):
    user_answer_str = str(user_answer)
    ideal_answer_str = str(ideal_answer)
    ideal_keywords = set(re.findall(r'\w+', ideal_answer_str.lower())) - stop_words
    user_keywords = set(re.findall(r'\w+', user_answer_str.lower())) - stop_words
    missing = ideal_keywords - user_keywords
    return sorted([word for word in missing if isinstance(word, str) and len(word) > 2])

# --- APPROACH 1 FEEDBACK LOGIC ---
def generate_rule_based_counterfactual(user_answer, negative_keywords, missing_keywords):
    if not negative_keywords or not missing_keywords:
        return None
    # The core of Approach 1: Arbitrary pairing
    term_to_replace = random.choice(negative_keywords)
    suggestion_term = random.choice(missing_keywords)
    # ... (logic to find sentence and generate suggestion) ...
    sentences = re.split(r'(?<=[.!?])\s+', user_answer)
    for sentence in sentences:
        if re.search(r'\b' + re.escape(term_to_replace) + r'\b', sentence, re.IGNORECASE):
            highlighted = re.sub(r'(\b' + re.escape(term_to_replace) + r'\b)', r'**\1**', sentence, flags=re.IGNORECASE)
            return f"For instance, in your sentence \"{highlighted}\", consider replacing '**{term_to_replace}**' with a concept like '**{suggestion_term}**'."
    return None

def generate_feedback_approach1(evaluation, user_answer, ideal_answer):
    feedback = ""
    if evaluation == 'correct':
        return "✅ Excellent! Your answer was comprehensive and accurate."
    
    negative_keywords = get_negative_shap_explanation(user_answer, ideal_answer)
    missing_keywords = get_missing_keywords(user_answer, ideal_answer)
    suggestion = generate_rule_based_counterfactual(user_answer, negative_keywords, missing_keywords)

    if suggestion:
        feedback += f"💡 **Suggestion for Improvement:**\n{suggestion}"
    else:
        feedback += f"❌ **To improve:** Your answer could be more complete by including key ideas like: {', '.join(missing_keywords[:5])}."
    return feedback

# --- 3. FLASK WEB ROUTES FOR DEMO ---
@app.route('/')
def index():
    # Select one random question for the demo
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
    feedback = generate_feedback_approach1(evaluation['evaluation'], user_answer, question_doc['Ideal Answer'])
    
    return render_template('demo_feedback.html', 
                           question=question_doc['Question'],
                           user_answer=user_answer,
                           evaluation=evaluation['evaluation'],
                           feedback=feedback,
                           approach_name="Approach 1: Rule-Based Counterfactual")

if __name__ == '__main__':
    app.run(debug=True, port=5004)
