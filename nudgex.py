import streamlit as st
from nudge_engine import process_trigger

# Set the title of the Streamlit app
st.title('NudgeX: Smart Nudge Factory')

# Define the list of available triggers
triggers = ["New User", "Long User Inactivity", "New Assets Added", "Used, Not Rated", "Near Top Contributor", "Used, No Feedback",
            "Used An Asset", "Used New Asset", "Negative Feedback Given", "Frequent Portal Usage", "New Asset, No Feedback"]

# Create a text input for the user ID
selected_user_id = st.text_input('Enter User ID', value="3000")

# Create a dropdown menu for selecting the trigger
selected_trigger = st.selectbox('Select a Trigger', triggers)

# Create a button to generate the nudge
if st.button('Generate Nudge'):
    # Get the user ID and convert to integer
    selected_user_id = int(selected_user_id)

    # Call the process_trigger function to generate the nudge
    nudge, explanation, accuracy, f1 = process_trigger(
        selected_user_id, selected_trigger)

    # Display the nudge, explanation, accuracy, and F1 score
    st.markdown(f'**Nudge:** {nudge}')
    st.markdown(f'**Explanation:** {explanation}')
    st.markdown(f'**Model Accuracy:** {accuracy}')
    st.markdown(f'**Model F1 Score:** {f1}')
