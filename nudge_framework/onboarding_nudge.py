# onboarding_nudge.py
# Author: Ganesh Subramanian
# Creation Date: 2023-08-12
# License: MIT
# This module is responsible for generating onboarding nudges for users.

import os
import joblib
import pandas as pd
import numpy as np
import lime.lime_tabular
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import streamlit as st
from datetime import datetime

# Function to assign a category to a user based on their interaction history

# Placeholder for trace messages
trace = st.sidebar.empty()

# This function writes a trace message to the Streamlit sidebar with a timestamp


def write_trace(msg):
    global trace
    timestamp = datetime.now().strftime("%H:%M:%S")
    trace = st.sidebar.text(f"{timestamp}: {msg}")


def assign_user_category(row):
    # write_trace("Onboarding | Assigning user category....")
    if pd.isnull(row['interaction_date']):  # No interaction history
        return 'New'
    elif (row['last_inactive_date'] - row['interaction_date']).days <= 7:  # Active in last 7 days
        return 'Red'
    else:  # Inactive for more than 7 days
        return 'Blue'


# Function to load and preprocess data
def load_and_preprocess_data():
    write_trace("Onboarding | Loading and preprocessing data....")
    # Load the data
    user_data = pd.read_csv('data/user_platform_interactions.csv')

    # Convert dates to datetime format
    user_data['interaction_date'] = pd.to_datetime(
        user_data['interaction_date'], format='%d-%m-%Y')
    user_data['last_inactive_date'] = pd.to_datetime(
        user_data['last_inactive_date'], format='%d-%m-%Y')

    # Calculate the number of days since the last activity
    user_data['days_since_last_activity'] = (pd.to_datetime(
        'now') - user_data['last_inactive_date']).dt.days

    # Extract the year, month, and day from the interaction date and last inactive date
    user_data['interaction_year'] = user_data['interaction_date'].dt.year
    user_data['interaction_month'] = user_data['interaction_date'].dt.month
    user_data['interaction_day'] = user_data['interaction_date'].dt.day
    user_data['last_inactive_year'] = user_data['last_inactive_date'].dt.year
    user_data['last_inactive_month'] = user_data['last_inactive_date'].dt.month
    user_data['last_inactive_day'] = user_data['last_inactive_date'].dt.day

    # Assign a category to each user
    user_data['user_category'] = user_data.apply(assign_user_category, axis=1)

    # One-hot encode the activity column
    user_data = pd.get_dummies(
        user_data, columns=['activity'], drop_first=True)

    # Drop the original date columns
    user_data = user_data.drop(
        columns=['interaction_date', 'last_inactive_date'])

    return user_data


# Function to train the classifier
def train_classifier(user_data):
    write_trace("Onboarding | Training classifier....")
    # Separate the features and the target variable
    X = user_data.drop(['userId', 'user_category'], axis=1)
    y = user_data['user_category']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Define the path to the model
    model_dir = "models"
    model_file = "onboarding_nudge_model.pkl"
    model_path = os.path.join(model_dir, model_file)

    # Create the directory if it doesn't exist
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # Load the model if it exists, otherwise train a new one
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
    else:
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        joblib.dump(clf, model_path)

    return clf, X_train, X_test, y_train, y_test


# Function to classify a user
def classify_user(user_id, clf, user_data):
    write_trace("Onboarding | Classifying user....")
    # Convert the user_id to integer
    user_id = int(user_id)

    # Check if the user_id exists in the data
    if user_id in user_data['userId'].values:
        # Get the row corresponding to the user_id
        user_row = user_data[user_data['userId'] == user_id]

        # Get the features of the user
        user_features = user_row.drop(
            ['userId', 'user_category'], axis=1).values

        # Predict the category of the user
        user_category = clf.predict(user_features)[0]
    else:
        user_category = 'New'  # If user_id not found in data, treat as new user

    return user_id, user_category


# Function to generate a nudge based on the user category
def generate_nudge(user_category, trigger):
    write_trace("Onboarding | Generating nudge....")
    # Define the nudge for each category
    if user_category == 'Red':
        nudge = 'Welcome Back! Start by exploring popular assets in our repository.'
    elif user_category == 'Blue':
        nudge = 'Did you know you can rate assets? Try it out on an asset you\'ve used recently.'
    else:  # New user
        nudge = 'Complete your profile to get personalized asset recommendations.'

    return nudge


# Function to explain the classification
def explain_classification(user_id, clf, explainer, user_data, feature_names):
    write_trace("Onboarding | Explaining classification....")
    # Convert the user_id to integer
    user_id = int(user_id)

    # Check if the user_id exists in the data
    if user_id in user_data['userId'].values:
        # Get the instance corresponding to the user_id
        instance = user_data.loc[user_data['userId']
                                 == user_id, feature_names].values[0]

        # Explain the instance
        exp = explainer.explain_instance(
            instance, clf.predict_proba, num_features=5)

        # Get the explanation as a list
        explanation = exp.as_list()
    else:  # New user
        explanation = [("User ID not found in data", "Treated as new user")]

    return explanation


# Function to format the explanation
def format_explanation(explanation):
    write_trace("Onboarding | Formatting explanation....")
    formatted_explanation = "The classification was primarily influenced by the following conditions:\n"
    for condition, weight in explanation:
        if isinstance(weight, str):  # if weight is a string, print it as is
            formatted_explanation += f"  - {condition}, with a detail: {weight}\n"
        else:  # if weight is a number, print it with 3 decimal places
            formatted_explanation += f"  - {condition}, with a weight of {weight:.3f}\n"
    return formatted_explanation


# Function to generate an onboarding nudge for a user
def generate_onboarding_nudge(user_id):
    write_trace("Onboarding | Generating onboarding nudge....")
    # Load and preprocess the data
    user_data = load_and_preprocess_data()

    # Train the classifier
    clf, X_train, X_test, y_train, y_test = train_classifier(user_data)

    # Classify the user
    _, user_category = classify_user(user_id, clf, user_data)

    # Generate the nudge
    nudge = generate_nudge(user_category, "Signup-Login")

    # Get the feature names
    feature_names = user_data.drop(
        ['userId', 'user_category'], axis=1).columns.tolist()

    # Create a LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(user_data[feature_names].values, feature_names=feature_names, class_names=[
                                                       'Red', 'Blue', 'New'], discretize_continuous=True)

    # Explain the classification
    explanation = explain_classification(
        user_id, clf, explainer, user_data, feature_names)

    # Format the explanation
    formatted_explanation = format_explanation(explanation)

    # Predict the class labels for the test set
    y_pred = clf.predict(X_test)

    # Calculate the accuracy and F1 score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    write_trace("Onboarding | Onboarding nudge generated.")

    return nudge, formatted_explanation, accuracy, f1
