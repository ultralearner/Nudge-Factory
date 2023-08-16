# feedback_nudge.py
# Author: Ganesh Subramanian
# Creation Date: 2023-08-12
# License: MIT
# Description: Module for generating feedback nudges using the RandomForest classifier.

import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import os


def write_trace(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp}: {msg}")


def preprocess_feedback(feedback):
    feedback = feedback.lower()
    feedback = re.sub(r'\d+', '', feedback)
    feedback = re.sub(r'\W+', ' ', feedback)
    feedback = feedback.strip()
    return feedback


def format_explanation(explanation):
    explanation_list = explanation.as_list()
    explanation_str = ", ".join(
        [f"'{word}' ({round(weight, 3)})" for word, weight in explanation_list])
    explanation_sentence = f"The sentiment prediction is primarily influenced by the following words: {explanation_str}."
    return explanation_sentence


def train_model(df_ratings):
    write_trace("Feedback | Performing sentiment analysis....")

    tfidf = TfidfVectorizer()
    svm = LinearSVC(dual=True)

    # Calibrate the SVM to enable probability predictions
    calibrated_svm = CalibratedClassifierCV(svm)

    pipeline = make_pipeline(tfidf, calibrated_svm)

    X = df_ratings['verbose_feedback'].apply(preprocess_feedback)
    y = df_ratings['sentiment']
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # Save the model and necessary objects
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(pipeline, 'models/sentiment_classifier.pkl')
    joblib.dump(tfidf, 'models/tfidf.pkl')
    joblib.dump(le, 'models/le.pkl')

    return X_test, y_test


def generate_feedback_nudge(user_id, trigger):
    nudge = None
    explanation_sentence = None
    accuracy = None
    f1 = None

    try:
        # Load the saved model and necessary objects
        pipeline = joblib.load('models/sentiment_classifier.pkl')
        tfidf = joblib.load('models/tfidf.pkl')
        le = joblib.load('models/le.pkl')

        write_trace("Feedback | Loading user feedback....")
        user_feedback = df_ratings[df_ratings['userId']
                                   == user_id]['verbose_feedback']
        user_feedback = user_feedback.apply(preprocess_feedback)
        sentiment_prediction = pipeline.predict([user_feedback.iloc[0]])

        explainer = LimeTextExplainer(
            class_names=["negative", "neutral", "positive"])

        explanation = explainer.explain_instance(
            user_feedback.iloc[0], pipeline.predict_proba, num_features=5)

        explanation_sentence = format_explanation(explanation)

        # Compute the accuracy and F1 score
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        if trigger == "Negative Feedback Given":
            nudge = "We're sorry to hear that. We're constantly working on improving our assets. Your feedback helps!"
        elif trigger == "Positive Feedback":
            nudge = "We're glad to hear you had a great experience! Your feedback helps us continue to provide high-quality assets."
        elif trigger == "Used An Asset" or trigger == "Used, No Feedback":
            nudge = "Your input is important to us. Could you please leave feedback for the assets you've used?"
        elif trigger == "Used New Asset" or trigger == "New Asset, No Feedback":
            nudge = "We noticed you've used a new asset. We'd love to hear your thoughts on it!"
        elif trigger == "Frequent Portal Usage":
            nudge = "As a frequent user, your feedback is invaluable in helping us improve our offerings."
        elif trigger == "Infrequent Portal Usage":
            nudge = "We noticed you haven't been around much. We'd appreciate your feedback on how we can improve your experience."

        return nudge, explanation_sentence, accuracy, f1

    except Exception as e:
        write_trace(f"An error occurred: {str(e)}")
        return nudge, explanation_sentence, accuracy, f1


# Load the data once outside the function
df_ratings = pd.read_csv(
    'data/master_dataset.csv')

# Train the model once outside the function
X_test, y_test = train_model(df_ratings)
