

# AI-Powered Interview Coach with Hybrid Feedback System

This project is a sophisticated, AI-powered interview coaching platform designed to provide users with a realistic interview experience and generate detailed, actionable feedback. Unlike traditional platforms that offer generic advice, this system uses a hybrid approach, combining a machine learning classifier for objective evaluation with a Large Language Model (LLM) for nuanced, human-like feedback generation.

The core of the project is its research into different feedback methodologies, moving from simple rule-based suggestions to a final, integrated system that leverages the reasoning capabilities of modern LLMs to provide specific, contextual, and encouraging feedback.

## Core Features

  * **CV-Driven Personalization:** The interview begins by analyzing a user's uploaded CV (PDF or TXT) to extract key technical skills.
  * **Dynamic Question Selection:** It tailors the interview by selecting relevant questions from a MongoDB database based on the skills identified in the CV.
  * **Hybrid Answer Evaluation:** A custom-trained **CatBoost classifier** evaluates user answers, classifying them as 'correct', 'partially correct', or 'incorrect'. The model uses a combination of semantic, lexical, and quantitative features:
      * **Semantic Similarity:** Cosine similarity between user and ideal answer embeddings (using `Sentence-Transformers`).
      * **Keyword Overlap:** Jaccard similarity to measure shared keywords.
      * **Length Analysis:** The difference in word count between the user's and the ideal answer.
  * **Explainable AI (XAI) for Keyword Identification:** Uses **SHAP (SHapley Additive exPlanations)** to identify specific keywords that positively or negatively influenced the model's evaluation.
  * **Advanced Feedback Generation:** The primary research component, exploring multiple strategies for generating constructive feedback (see details below).
  * **Interactive Chat Interface:** A web-based chatbot interface built with Flask provides a seamless and engaging user experience.

## The Research: Feedback Generation Approaches

This project systematically explores and builds upon three primary methodologies for generating feedback, each with increasing sophistication.

### Approach 1: Rule-Based & Semantic Counterfactuals

This approach focuses on generating "counterfactuals"—specific suggestions on what to change in an answer to improve it.

  * **`demo_approach_1.py` (Rule-Based):**
      * **Identify Flaws:** Uses SHAP to find `negative_keywords` that contributed to a low score.
      * **Identify Gaps:** Determines `missing_keywords` by comparing the user's answer to the ideal answer.
      * **Generate Suggestion:** **Randomly** pairs a negative keyword with a missing keyword and suggests replacing one with the other.
  * **`demo_approach_1_enhanced.py` (Semantic-Based):**
      * This enhancement improves upon the random pairing.
      * It uses **Sentence-Transformers** to find the most *semantically similar* `missing_keyword` for a given `negative_keyword`.
      * This results in a more logical and contextually relevant suggestion for replacement.

### Approach 2: LLM-Enhanced Summaries

This approach leverages a high-speed LLM (Groq with LLaMA 3.1) to create more descriptive and encouraging feedback.

  * **`demo_approach_2.py` (LLM Keyword Summary):**
      * **Identify Strengths:** For 'partially correct' answers, it uses SHAP to find `positive_keywords`.
      * **Summarize Strengths:** The LLM receives these keywords and generates a concise, human-readable sentence summarizing what the user did well.
      * **Identify Gaps:** It lists the key `missing_keywords` for the user to include.
  * **`demo_approach_2_enhanced.py` (LLM Explanation):**
      * This enhancement goes beyond just listing missing keywords.
      * It tasks the LLM with explaining *why* the `missing_keywords` are important in the context of the ideal answer.
      * This provides deeper conceptual understanding rather than just a checklist of terms.

### Approach 3: The Integrated Hybrid System (`app_full.py`)

This is the final, deployed version within the main application, combining the best elements of the previous approaches into a cohesive and powerful feedback engine.

1.  **Strength Summarization:** Uses the LLM summary from Approach 2 for 'partially correct' answers.
2.  **Intelligent Counterfactuals:** Employs the LLM to analyze both `negative_keywords` and `missing_keywords` to find the most **logically sound pairs** for counterfactual suggestions. This is a more advanced version of the semantic pairing in Approach 1.
3.  **Conceptual Explanations:** Incorporates the LLM-generated explanations of missing concepts from Approach 2 Enhanced.
4.  **Comprehensive Report:** Combines these elements into a final feedback report that praises strengths, offers specific, actionable suggestions for improvement, and explains the underlying concepts for deeper learning.

## System Architecture

The application is built on a modern web stack, integrating ML models, an external LLM API, and a NoSQL database.

<img width="1633" height="1121" alt="research drawio" src="https://github.com/user-attachments/assets/40249ba9-cadc-4974-8801-160f9c58d9f9" />




## Tech Stack

  * **Backend:** Flask
  * **Machine Learning:** CatBoost, Scikit-learn, SHAP
  * **NLP & Semantic Search:** Sentence-Transformers, NLTK
  * **LLM Provider:** Groq (LLaMA 3.1 8B Instant)
  * **Database:** MongoDB
  * **Deployment & Tooling:** Python, Joblib, PyPDF2, python-dotenv

## File Structure

```
.
├── app_full.py                 # Main Flask application with the integrated feedback system.
├── demo_approach_1.py          # Demo for the Rule-Based Counterfactual approach.
├── demo_approach_1_enhanced.py # Demo for the Semantic Counterfactual enhancement.
├── demo_approach_2.py          # Demo for the LLM Keyword Summary approach.
├── demo_approach_2_enhanced.py # Demo for the LLM Missing Concept Explanation enhancement.
├── catboost_classifier.cbm     # Trained CatBoost model for answer evaluation.
├── label_encoder.pkl           # Saved Scikit-learn LabelEncoder for class names.
├── requirements.txt            # Python dependencies.
├── .env                        # Local environment variables (needs to be created).
└── templates/
    ├── demo_feedback.html      # Template for the demo feedback pages.
    ├── demo_interview.html     # Template for the single-question demo page.
    ├── feedback.html           # Final feedback report page for the full app.
    ├── index.html              # CV upload page for the full app.
    └── interview.html          # Chatbot interface for the full app.
```

