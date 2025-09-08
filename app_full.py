# app_full.py
import os
import pandas as pd
import numpy as np
import re
import shap
import json
import random
from groq import Groq # Import the Groq library
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from catboost import CatBoostClassifier
import nltk
from nltk.corpus import stopwords
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import PyPDF2
import io
from pymongo import MongoClient

# --- 1. INITIAL SETUP & MODEL LOADING ---
print("--- ⏳ Initializing & Loading Models (Groq LLaMA3 Version) ---")
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Groq API Configuration ---
try:
    # IMPORTANT: Make sure your .env file has GROQ_API_KEY="your_key_here"
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file.")
    client_groq = Groq(api_key=api_key)
    LLM_MODEL = "llama-3.1-8b-instant" # Using LLaMA3 8B model on Groq
    print(f"✅ Groq API client configured with model: {LLM_MODEL}")
except Exception as e:
    print(f"❌ Groq API Key Error: {e}")
    exit()

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
if not MONGO_URI:
    raise Exception("MONGO_CONNECTION_STRING not found in .env file.")
DB_NAME = "InterviewBotDB"
COLLECTION_NAME = "questions"

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    questions_collection = db[COLLECTION_NAME]
    print("✅ Successfully connected to MongoDB.")
except Exception as e:
    print(f"❌ Could not connect to MongoDB: {e}")
    exit()

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

try:
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    classifier = CatBoostClassifier()
    classifier.load_model("catboost_classifier.cbm")
    label_encoder = joblib.load('label_encoder.pkl')
    print("✅ All ML models loaded.")
except FileNotFoundError as e:
    print(f"❌ Model loading error: {e}. Run train.py first.")
    exit()

# --- 2. CORE LOGIC FUNCTIONS ---
def extract_text_from_cv(file_storage):
    if file_storage.filename.endswith('.pdf'):
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_storage.read()))
            text = "".join(page.extract_text() for page in pdf_reader.pages)
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None
    elif file_storage.filename.endswith('.txt'):
        return file_storage.read().decode('utf-8')
    return None

def get_info_from_cv_with_groq(cv_text):
    prompt = f"""
    Analyze the following CV text. Extract the candidate's full name and a list of their technical skills (e.g., Python, Java, SQL, React, AWS, Docker).
    Based on their name, generate a friendly, one-sentence greeting to start an interview.
    Return the information as a single JSON object with three keys: "name", "skills", and "greeting".
    ---
    {cv_text}
    ---
    """
    try:
        chat_completion = client_groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=LLM_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        response_json = json.loads(chat_completion.choices[0].message.content)
        return response_json
    except Exception as e:
        print(f"Groq CV analysis failed: {e}")
        return None

def select_questions(skills, num_questions=3):
    skill_category_map = {
        'java': 'Java', 'python': 'Languages and Frameworks', 'sql': 'Database and SQL',
        'javascript': 'Web Development', 'react': 'Web Development', 'php': 'PHP',
        'git': 'Version Control', 'docker': 'DevOps', 'kubernetes': 'DevOps',
        'aws': 'DevOps', 'azure': 'DevOps', 'oop': 'General Programming',
        'data structures': 'Data Structures', 'algorithms': 'Data Structures'
    }
    relevant_categories = {skill_category_map.get(skill.lower()) for skill in skills if skill.lower() in skill_category_map}
    query = {"Category": {"$in": list(relevant_categories)}}
    relevant_docs = list(questions_collection.find(query, {"_id": 0}))
    if len(relevant_docs) < num_questions:
        needed = num_questions - len(relevant_docs)
        general_query = {"Category": "General Programming"}
        existing_questions = {doc['Question'] for doc in relevant_docs}
        general_docs = list(questions_collection.find(general_query, {"_id": 0}))
        new_general_docs = [doc for doc in general_docs if doc['Question'] not in existing_questions]
        if len(new_general_docs) >= needed:
            relevant_docs.extend(random.sample(new_general_docs, needed))
        else:
            relevant_docs.extend(new_general_docs)
    random.shuffle(relevant_docs)
    return relevant_docs[:num_questions]

# --- 3. EVALUATION & FEEDBACK FUNCTIONS (APPROACH 3) ---

def preprocess_text(text):
    if not isinstance(text, str): text = str(text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

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

def get_logical_pairs_with_groq(user_answer, negative_keywords, missing_keywords):
    negative_str = ", ".join(negative_keywords)
    missing_str = ", ".join(missing_keywords)
    prompt = f"""
    Analyze the user's answer, a list of negative keywords from their answer, and a list of missing keywords from the ideal answer.
    User Answer: "{user_answer}"
    Negative Keywords: [{negative_str}]
    Missing Keywords: [{missing_str}]
    Your task is to find up to 2 of the most logical pairs where a missing keyword is a direct conceptual replacement for a negative keyword. Only create a pair if it makes strong logical sense for improving the answer.
    Return the result as a JSON object with a single key "pairs" which contains a list of objects. If no logical pairs are found, return an empty list.
    Example: {{"pairs": [{{"term_to_replace": "word1", "suggestion_term": "word2"}}]}}
    """
    try:
        chat_completion = client_groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=LLM_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        response_json = json.loads(chat_completion.choices[0].message.content)
        return response_json.get("pairs", [])
    except Exception as e:
        print(f"Groq logical pair generation failed: {e}")
        return []

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
            messages=[{"role": "user", "content": prompt}],
            model=LLM_MODEL,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq strengths summary failed: {e}")
        return None

def explain_missing_concepts_with_groq(ideal_answer, missing_keywords):
    if not missing_keywords: return None
    missing_str = ", ".join(missing_keywords)
    prompt = f"""
    A user's answer was missing the following key concepts: [{missing_str}].
    Based on the ideal answer provided below, write a short, helpful explanation (2-3 sentences) for a student, explaining why these concepts are important for a complete answer. Do not just list the keywords; explain their role.
    Ideal Answer: "{ideal_answer}"
    """
    try:
        chat_completion = client_groq.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=LLM_MODEL,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq missing concepts explanation failed: {e}")
        return None

def generate_counterfactual_suggestions(user_answer, keyword_pairs):
    suggestions = []
    sentences = re.split(r'(?<=[.!?])\s+', user_answer)
    used_sentences = set()
    for pair in keyword_pairs:
        term_to_replace, suggestion_term = pair.get("term_to_replace"), pair.get("suggestion_term")
        if not term_to_replace or not suggestion_term: continue
        for sentence in sentences:
            if sentence not in used_sentences and re.search(r'\b' + re.escape(term_to_replace) + r'\b', sentence, re.IGNORECASE):
                highlighted = re.sub(r'(\b' + re.escape(term_to_replace) + r'\b)', r'**\1**', sentence, flags=re.IGNORECASE)
                suggestions.append(f"In your sentence \"{highlighted}\", replacing '**{term_to_replace}**' with a concept like '**{suggestion_term}**' would make your answer more precise.")
                used_sentences.add(sentence)
                break
    return suggestions

def generate_final_feedback(evaluation, user_answer, ideal_answer, positive_keywords, counterfactual_suggestions, missing_concepts_explanation):
    feedback = ""
    if evaluation == 'partially correct':
        feedback += "This is a good answer that is on the right track.\n"
        strengths_summary = summarize_strengths_with_groq(user_answer, positive_keywords)
        if strengths_summary:
            feedback += f"\n✅ **What you did well:** {strengths_summary}\n"
    elif evaluation == 'correct':
        return "✅ Excellent! Your answer was comprehensive and accurate."
    else: # incorrect
        feedback += "It seems your answer missed the key concepts for this question.\n"

    improvement_feedback = ""
    if counterfactual_suggestions:
        improvement_feedback += "\n💡 **Specific Suggestions:**\n" + "\n".join([f"- {s}" for s in counterfactual_suggestions])
    
    if missing_concepts_explanation:
        improvement_feedback += f"\n\n🧠 **Key Concepts to Review:**\n{missing_concepts_explanation}"

    if not improvement_feedback:
        missing_keywords = get_missing_keywords(user_answer, ideal_answer)
        if missing_keywords:
             improvement_feedback += f"\n❌ **To improve:** Your answer could be more complete by including ideas like: **{', '.join(missing_keywords[:5])}**."

    return feedback + improvement_feedback

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
    score = 2 if evaluation_result == 'correct' else (1 if evaluation_result == 'partially correct' else 0)
    return {"evaluation": evaluation_result, "score": score}

# --- 4. FLASK WEB ROUTES (MODIFIED FOR CHATBOT) ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('cv_file')
        if not file or file.filename == '':
            return render_template('index.html', error="No file selected.")
        cv_text = extract_text_from_cv(file)
        if not cv_text:
            return render_template('index.html', error="Could not read file. Please use PDF or TXT.")
        cv_info = get_info_from_cv_with_groq(cv_text) # Use Groq for CV parsing
        if not cv_info:
            return render_template('index.html', error="Could not analyze CV with AI.")
        
        session['interview_data'] = {
            "name": cv_info['name'], "greeting": cv_info['greeting'],
            "skills": cv_info['skills'],
            "questions": select_questions(cv_info['skills']), "answers": [], "current_q_index": 0
        }
        return redirect(url_for('interview_page'))
    return render_template('index.html')

@app.route('/interview')
def interview_page():
    if 'interview_data' not in session:
        return redirect(url_for('index'))
    return render_template('interview.html')

# --- NEW API ENDPOINTS FOR THE CHATBOT ---

@app.route('/get-chat-message', methods=['GET'])
def get_chat_message():
    if 'interview_data' not in session:
        return jsonify({"error": "Interview session not found."}), 404

    interview_data = session['interview_data']
    q_index = interview_data['current_q_index']
    questions = interview_data['questions']

    if q_index >= len(questions):
        return jsonify({"message": "Interview complete!", "finished": True})

    question_text = questions[q_index]['Question']
    response_data = {"message": f"<strong>Question {q_index + 1}:</strong><br>{question_text}", "finished": False}
    
    if q_index == 0:
        greeting = interview_data.get('greeting', 'Hello!')
        skills = interview_data.get('skills', [])
        skills_html = "".join([f'<span class="badge bg-secondary me-1 mb-1">{skill}</span>' for skill in skills])
        full_greeting = (
            f"{greeting}<br><br>"
            f"<div class='p-3 bg-light rounded border'>"
            f"<p class='mb-2'><strong>Based on your CV, I've identified the following skills and will be asking questions related to them:</strong></p>"
            f"<div>{skills_html}</div></div><br>"
            f"Let's begin. Your first question is:<br><br><strong>{question_text}</strong>"
        )
        response_data["message"] = full_greeting

    return jsonify(response_data)

@app.route('/submit-chat-message', methods=['POST'])
def submit_chat_message():
    if 'interview_data' not in session:
        return jsonify({"error": "Interview session not found."}), 404

    data = request.get_json()
    user_answer = data.get('answer')
    if not user_answer:
        return jsonify({"error": "No answer provided."}), 400

    interview_data = session['interview_data']
    q_index = interview_data['current_q_index']
    questions = interview_data['questions']
    current_question_doc = questions[q_index]

    evaluation = run_evaluation(user_answer, current_question_doc['Ideal Answer'])
    
    interview_data['answers'].append({
        "question": current_question_doc['Question'], "answer": user_answer,
        "evaluation": evaluation['evaluation'], "score": evaluation['score'],
        "ideal_answer": current_question_doc['Ideal Answer']
    })
    interview_data['current_q_index'] += 1
    session['interview_data'] = interview_data

    return jsonify({"success": True})

# --- FEEDBACK ROUTE (NOW USES APPROACH 3 with Groq) ---
@app.route('/feedback')
def feedback():
    if 'interview_data' not in session: return redirect(url_for('index'))
    results = session.get('interview_data', {}).get('answers', [])
    name = session.get('interview_data', {}).get('name', 'Candidate')

    print("--- ⏳ Generating Detailed Feedback for Final Report (Approach 3 with Groq) ---")
    for result in results:
        ideal_answer = result.get('ideal_answer')
        if ideal_answer:
            positive_keywords, counterfactual_suggestions, missing_concepts_explanation = [], [], None
            
            if result['evaluation'] == 'partially correct':
                positive_keywords = get_positive_shap_explanation(result['answer'], ideal_answer)

            if result['evaluation'] != 'correct':
                negative_keywords = get_negative_shap_explanation(result['answer'], ideal_answer)
                missing_keywords = get_missing_keywords(result['answer'], ideal_answer)
                
                if negative_keywords and missing_keywords:
                    logical_pairs = get_logical_pairs_with_groq(result['answer'], negative_keywords, missing_keywords)
                    counterfactual_suggestions = generate_counterfactual_suggestions(result['answer'], logical_pairs)
                
                if missing_keywords:
                    missing_concepts_explanation = explain_missing_concepts_with_groq(ideal_answer, missing_keywords)
            
            result['detailed_feedback'] = generate_final_feedback(
                result['evaluation'], result['answer'], ideal_answer,
                positive_keywords, counterfactual_suggestions, missing_concepts_explanation
            )

    total_score = sum(item['score'] for item in results)
    max_score = len(results) * 2
    final_percentage = (total_score / max_score) * 100 if max_score > 0 else 0
    session.pop('interview_data', None)
    return render_template('feedback.html', name=name, results=results,
                           total_score=total_score, max_score=max_score,
                           final_percentage=final_percentage)

if __name__ == '__main__':
    app.run(debug=True)
