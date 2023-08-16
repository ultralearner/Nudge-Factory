# Filename: User Trigger.py
# Author: Ganesh Subramanian
# Date Created: 2023-08-12
# License: MIT
# Description: Script for generating nudges based on user triggers in the NudgeX platform.

import streamlit as st
from nudge_engine import process_trigger, display_recommendations

# Path to images directory
image_path = "pics/"

# Display the banner and welcome message
st.image(image_path + "nudgex-banner.png")


# Subtitle for logs
st.sidebar.markdown("## Logs")

# Display login information
st.markdown("<h2 style='text-align: right; color: black;'>Logged in as Portal Admin</h2>",
            unsafe_allow_html=True)

# Define the list of available triggers for each category
triggers = {
    "Onboarding": ["Signup-Login"],
    "Engagement": ["Long User Inactivity", "Near Top Contributor", "New Assets Added", "Used, Not Rated", "Used, No Feedback"],
    "Feedback": ["Frequent Portal Usage", "Negative Feedback Given", "New Asset, No Feedback", "Used An Asset", "Used New Asset"]
}

# Define the list of categories
categories = list(triggers.keys())

# User inputs
# Text input for user ID
selected_user_id = st.text_input('Enter User ID', value="3000")
# Dropdown for category selection
selected_category = st.selectbox('Select a Category', categories)
# Dropdown for trigger selection
selected_trigger = st.selectbox(
    'Select a Trigger', triggers[selected_category])
# Text input for asset ID
selected_asset_id = st.text_input('Enter Asset ID', value="50")

# Button click action
if st.button('Generate Nudge'):
    # Get the user ID and asset ID and convert to integer
    selected_user_id = int(selected_user_id)
    selected_asset_id = int(selected_asset_id)

    # Call the process_trigger function to generate the nudge
    nudge, explanation, accuracy, f1, predicted_rating = process_trigger(
        selected_user_id, selected_asset_id, selected_trigger)

    # Display the recommendations
    recommendations = display_recommendations(selected_user_id)
    st.markdown(f'**Recommended Assets for User {selected_user_id}:**')
    buttons = ""
    for recommendation in recommendations:
        buttons += f'<button style="margin: 5px; padding: 10px; border: none; color: white; background-color: #0081AA;">{recommendation}</button> '
    st.markdown(buttons, unsafe_allow_html=True)

    st.markdown('---')

    # Display the nudge, explanation, accuracy, F1 score, predicted rating, and recommendations
    st.markdown(f'**Nudge:** {nudge}')
    st.markdown(f'**Explanation:** {explanation}')
    st.markdown(f'**Model Accuracy:** {accuracy}')
    st.markdown(f'**Model F1 Score:** {f1}')
    st.markdown(f'**Predicted Rating:** {float(predicted_rating):.1f}')

# Display the apps logos
st.image(image_path + "apps-logos.png")
