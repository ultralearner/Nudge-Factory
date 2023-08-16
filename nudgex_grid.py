import streamlit as st
from nudge_engine import process_trigger

# Set the title of the Streamlit app
st.title('NudgeX: Smart Nudge Factory')

# Define the list of available triggers
triggers = ["New User", "Long User Inactivity", "New Assets Added", "Used, Not Rated", "Near Top Contributor", "Used, No Feedback",
            "Used An Asset", "Used New Asset", "Negative Feedback Given", "Frequent Portal Usage", "New Asset, No Feedback"]

# Define the list of preconfigured user IDs
user_ids = ["3298", "3299", "3100", "3200"]

# Create a text input for the user ID
selected_user_id = st.text_input('Enter User ID', value="3298")

# Create a button to generate all nudges
if st.button('Generate All Nudges'):
    # Get the user ID and convert to integer
    selected_user_id = int(selected_user_id)

    # Check if the selected user ID is in the preconfigured list
    if selected_user_id not in user_ids:
        # If not, display a message indicating that the entered user ID will be used instead
        st.write(
            f"User ID {selected_user_id} is not in the preconfigured list. Using the entered User ID.")

    # Loop through each trigger and generate the nudge
    for trigger in triggers:
        nudge, explanation, accuracy, f1 = process_trigger(
            selected_user_id, trigger)
        # Display the trigger, nudge, explanation, accuracy, and F1 score
        st.markdown(f'**Trigger:** {trigger}')
        st.markdown(f'**Nudge:** {nudge}')
        st.markdown(f'**Explanation:** {explanation}')
        st.markdown(f'**Model Accuracy:** {accuracy}')
        st.markdown(f'**Model F1 Score:** {f1}')
        # Add a horizontal line to separate each nudge
        st.markdown('---')
