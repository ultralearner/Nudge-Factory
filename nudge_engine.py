# nudge_engine.py
# Author: Ganesh Subramanian
# Creation Date: 2023-08-12
# License: MIT
# Description: Module for processing triggers and generating nudges using the NudgeX framework.

from nudge_framework.onboarding_nudge import generate_onboarding_nudge
from nudge_framework.engagement_nudge import generate_engagement_nudge
from nudge_framework.feedback_nudge import generate_feedback_nudge
from nudge_framework.user_rating_prediction import predict_rating, load_rating_prediction_model
from nudge_framework.recommender_nudge import load_recommender_model, generate_recommendations
import streamlit as st

# Load the recommender model
tfidf, tfidf_matrix, cosine_sim, indices, svd = load_recommender_model()


def display_recommendations(user_id):
    """Generate and display recommendations for the given user."""
    recommendations = generate_recommendations(
        user_id, tfidf, tfidf_matrix, cosine_sim, indices, svd)
    return recommendations


def process_trigger(user_id, asset_id, trigger):
    """Process the given trigger and generate an appropriate nudge."""
    user_id = int(user_id)

    if trigger == "Signup-Login":
        nudge, explanation, accuracy, f1 = generate_onboarding_nudge(user_id)
    elif trigger in ["Long User Inactivity", "New Assets Added", "Used, Not Rated", "Near Top Contributor"]:
        nudge, explanation, accuracy, f1 = generate_engagement_nudge(
            user_id, trigger)
    elif trigger in ["Used, No Feedback", "Used An Asset", "Used New Asset", "Negative Feedback Given", "Frequent Portal Usage", "New Asset, No Feedback"]:
        nudge, explanation, accuracy, f1 = generate_feedback_nudge(
            user_id, trigger)
    else:
        return "Invalid trigger.", None, None, None, None

    predicted_rating = predict_rating(user_id, asset_id)

    if accuracy is not None:
        accuracy = round(accuracy, 3)

    if f1 is not None:
        f1 = round(f1, 3)

    return nudge, explanation, accuracy, f1, predicted_rating
