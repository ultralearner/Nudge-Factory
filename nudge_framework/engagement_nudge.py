# engagement_nudge.py
# Author: Ganesh Subramanian
# Creation Date: 2023-08-12
# License: MIT
# Description: Module for generating engagement nudges using the RandomForest classifier.

import pandas as pd
import os
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from lime.lime_tabular import LimeTabularExplainer
import streamlit as st
from datetime import datetime
import pickle

# Ignore DataConversionWarning
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Placeholder for trace messages
trace = st.sidebar.empty()

# This function writes a trace message to the Streamlit sidebar with a timestamp


def write_trace(msg):
    global trace
    timestamp = datetime.now().strftime("%H:%M:%S")
    trace = st.sidebar.text(f"{timestamp}: {msg}")

# This function loads and preprocesses data


def load_and_preprocess_data():
    write_trace("Engagement | Loading and preprocessing data....")
    df = pd.read_csv(os.path.join('data', 'user_platform_interactions.csv'))
    le = LabelEncoder()
    categorical_features = df.select_dtypes(
        include=['object']).columns.tolist()
    df[categorical_features] = df[categorical_features].apply(
        lambda col: le.fit_transform(col.astype(str)), axis=0)
    return df, le

# This function trains a RandomForest classifier


def train_classifier(df, target):
    # Define the path to the pickle file
    model_path = 'models/engagement_rf.pkl'

    # Prepare the training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop([target, 'userId'], axis=1), df[target], test_size=0.2, random_state=42)

    # Check if the pickle file already exists
    if os.path.exists(model_path):
        write_trace("Engagement | Loading the existing model....")
        # Load the existing model from the pickle file
        with open(model_path, 'rb') as f:
            rf = pickle.load(f)
    else:
        write_trace("Engagement | Training the model....")
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)

        # Create the directory if it doesn't exist
        if not os.path.isdir('models'):
            os.makedirs('models')

        # Save the trained model to a pickle file
        with open(model_path, 'wb') as f:
            pickle.dump(rf, f)

    return rf, X_train, X_test, y_train, y_test


# Function to format the explanation
def format_explanation(explanation):
    write_trace("Engagement | Formatting explanation....")
    formatted_explanation = "The classification was primarily influenced by the following conditions:\n"
    for condition, weight in explanation:
        if isinstance(weight, str):  # if weight is a string, print it as is
            formatted_explanation += f"  - {condition}, with a detail: {weight}\n"
        else:  # if weight is a number, print it with 3 decimal places
            formatted_explanation += f"  - {condition}, with a weight of {weight:.3f}\n"
    return formatted_explanation


# This function classifies a user and generates a nudge

def classify_user_and_generate_nudge(user_id, trigger, rf, df, le, target):
    write_trace(
        f"Engagement | Classifying & Generating nudge for user {user_id}....")
    if user_id not in df['userId'].values:
        nudge = "Is there something I can help you with?"
        explanation = [("User ID not found in data", "Treated as new user")]
    else:
        user_data = df[df['userId'] == user_id].drop(
            [target, 'userId'], axis=1).iloc[0]
        user_action = rf.predict(user_data.values.reshape(1, -1))[0]

        trigger_to_nudge_mapping = {
            "Long User Inactivity": "It's been a while since your last visit. Check out what's new in our repository.",
            "New Assets Added": "Don't miss out on the latest assets. Log in today.",
            "Used, Not Rated": "Help the community by rating the assets you've used.",
            "Near Top Contributor": "You're just one asset away from becoming a top contributor this month.",
            "Used, No Feedback": "Your opinion matters. Share your feedback on the assets you've used."
        }

        base_nudge = trigger_to_nudge_mapping.get(
            trigger, "Please check out the popular assets in our repository.")

        if user_action == 0:  # Ignored the nudge
            addl_msg = "(+++ User ignored the previous nudge +++) We miss you! Check out our new assets that might interest you."
        elif user_action == 1:  # Dismissed the nudge
            addl_msg = "(+++ User Dismissed the previous nudge +++) Keep going! Check out these additional resources."
        else:  # Interacted with the nudge
            addl_msg = "(+++ User interacted with the nudge +++) Great job! You are highly engaged. Here are some advanced assets for you."

        nudge = base_nudge + " " + addl_msg

        explainer = LimeTabularExplainer(df.drop([target, 'userId'], axis=1).values,
                                         feature_names=df.drop(
            [target, 'userId'], axis=1).columns,
            class_names=[
            'Ignored', 'Dismissed', 'Interacted'],
            discretize_continuous=True)
        exp = explainer.explain_instance(
            user_data.values, rf.predict_proba, num_features=5)
        explanation = exp.as_list()

    # Format the explanation
    formatted_explanation = format_explanation(explanation)

    return nudge, formatted_explanation


# This function generates an engagement nudge for a user


def generate_engagement_nudge(user_id, trigger):
    df, le = load_and_preprocess_data()
    target = 'activity'
    rf, X_train, X_test, y_train, y_test = train_classifier(df, target)
    rf = pickle.load(open('models/engagement_rf.pkl', 'rb'))
    nudge, explanation = classify_user_and_generate_nudge(
        user_id, trigger, rf, df, le, target)
    write_trace("Engagement | Testing the model....")
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    write_trace("Engagement | Computing F1 score....")
    f1 = f1_score(y_test, y_pred, average='weighted') if len(
        set(y_test)) > 2 else f1_score(y_test, y_pred, average='binary')
    return nudge, explanation, accuracy, f1
