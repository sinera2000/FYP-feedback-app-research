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
from flask import Flask, render_template, request, session
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from bson import ObjectId # Import ObjectId to check for it
import json
import torch # Add this import for tensor operations

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

# --- ORIGINAL APPROACH 1 FEEDBACK LOGIC ---
def generate_rule_based_counterfactual(user_answer, negative_keywords, missing_keywords):
    if not negative_keywords or not missing_keywords:
        return None
    term_to_replace = random.choice(negative_keywords)
    suggestion_term = random.choice(missing_keywords)
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

# --- ENHANCED APPROACH 1: SEMANTIC COUNTERFACTUAL ---
def generate_semantic_counterfactual(user_answer, negative_keywords, missing_keywords, sbert_model):
    """
    Finds the most logical counterfactual pair by using semantic similarity.
    It compares the most significant negative keyword to all missing keywords
    and chooses the one with the highest cosine similarity.
    """
    if not negative_keywords or not missing_keywords:
        return None
    try:
        negative_embedding = sbert_model.encode(negative_keywords[0])
        missing_embeddings = sbert_model.encode(missing_keywords)
        similarities = util.cos_sim(negative_embedding, missing_embeddings)
        best_match_index = torch.argmax(similarities)
        term_to_replace = negative_keywords[0]
        suggestion_term = missing_keywords[best_match_index.item()]
        sentences = re.split(r'(?<=[.!?])\s+', user_answer)
        for sentence in sentences:
            if re.search(r'\b' + re.escape(term_to_replace) + r'\b', sentence, re.IGNORECASE):
                highlighted = re.sub(r'(\b' + re.escape(term_to_replace) + r'\b)', r'**\1**', sentence, flags=re.IGNORECASE)
                return f"For instance, in your sentence \"{highlighted}\", a more precise concept to use instead of '**{term_to_replace}**' would be '**{suggestion_term}**'."
    except Exception as e:
        print(f"Error during semantic pairing: {e}")
        return f"To improve, consider replacing concepts like '**{negative_keywords[0]}**' with key ideas such as: **{', '.join(missing_keywords[:3])}**."
    return None

def generate_feedback_approach1_enhanced(evaluation, user_answer, ideal_answer, sbert_model):
    """
    A new version of the feedback generator that uses the semantic counterfactual method.
    """
    feedback = ""
    if evaluation == 'correct':
        return "✅ Excellent! Your answer was comprehensive and accurate."
    negative_keywords = get_negative_shap_explanation(user_answer, ideal_answer)
    missing_keywords = get_missing_keywords(user_answer, ideal_answer)
    suggestion = generate_semantic_counterfactual(user_answer, negative_keywords, missing_keywords, sbert_model)
    if suggestion:
        feedback += f"💡 **Suggestion for Improvement:**\n{suggestion}"
    elif missing_keywords:
        feedback += f"❌ **To improve:** Your answer could be more complete by including key ideas like: **{', '.join(missing_keywords[:5])}**."
    else:
        feedback += "Your answer is on the right track, but could be more detailed."
    return feedback

# --- 3. FLASK WEB ROUTES FOR DEMO ---
@app.route('/')
def index():
    question_doc = list(questions_collection.aggregate([{"$sample": {"size": 1}}]))[0]
    if '_id' in question_doc:
        question_doc['_id'] = str(question_doc['_id'])
    session['question_doc'] = question_doc
    return render_template('demo_interview.html', question=question_doc['Question'])

@app.route('/submit', methods=['POST'])
def submit():
    user_answer = request.form['user_answer']
    question_doc = session['question_doc']
    
    evaluation = run_evaluation(user_answer, question_doc['Ideal Answer'])
    # Calling the new, enhanced function
    feedback = generate_feedback_approach1_enhanced(evaluation['evaluation'], user_answer, question_doc['Ideal Answer'], sbert_model)
    
    return render_template('demo_feedback.html', 
                           question=question_doc['Question'],
                           user_answer=user_answer,
                           evaluation=evaluation['evaluation'],
                           feedback=feedback,
                           approach_name="Approach 1: Rule-Based Counterfactual (Enhanced)")

if __name__ == '__main__':
    app.run(debug=True, port=5004)

