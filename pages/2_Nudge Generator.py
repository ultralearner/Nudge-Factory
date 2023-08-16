import streamlit as st
from nudge_engine import process_trigger, display_recommendations


# add logo of the portal Nudgex from the pics folder on the top
st.image("pics/nudgex-banner.png")

# draw a divider line
st.sidebar.markdown("---")

# display a subtitle as "Logs"
st.sidebar.markdown("## Logs")

# st.markdown(
#     '<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
st.markdown("<h5 style='text-align: right; color: black;'>Logged in as Portal Admin</h5>",
            unsafe_allow_html=True)


# Define the list of available triggers for each category
triggers = {
    "Onboarding": ["Signup-Login"],
    "Engagement": ["Long User Inactivity", "Near Top Contributor", "New Assets Added", "Used, Not Rated", "Used, No Feedback"],
    "Feedback": ["Frequent Portal Usage", "Negative Feedback Given", "New Asset, No Feedback", "Used An Asset", "Used New Asset"]
}

# Define the list of preconfigured user IDs
user_ids = ["3298", "3299", "3100", "3200"]

# Create a text input for the user ID
selected_user_id = st.text_input('Enter User ID', value="3298")

# Create a text input for the asset ID
selected_asset_id = st.text_input('Enter Asset ID', value="50")

# Create a button to generate all nudges
if st.button('Generate All Nudges'):
    # Get the user ID and convert to integer
    selected_user_id = int(selected_user_id)

    # Display the recommendations
    recommendations = display_recommendations(selected_user_id)
    # Display the recommendations
    st.markdown(f'**Recommended Assets for User {selected_user_id}:**')
    buttons = ""
    for recommendation in recommendations:
        buttons += f'<button style="margin: 5px; padding: 10px; border: none; color: white; background-color: #0081AA;">{recommendation}</button> '
    st.markdown(buttons, unsafe_allow_html=True)

    st.markdown('---')

    # Loop through each category and trigger and generate the nudge
    for category, category_triggers in triggers.items():
        st.markdown(f'## {category}')  # Display the category as a subheading
        for trigger in category_triggers:
            nudge, explanation, accuracy, f1, predicted_rating = process_trigger(
                selected_user_id, selected_asset_id, trigger)
            # Display the trigger, nudge, explanation, accuracy, and F1 score
            st.markdown(f'**Trigger:** {trigger}')
            st.markdown(f'**Nudge:** {nudge}')
            st.markdown(f'**Explanation:** {explanation}')
            st.markdown(f'**Model Accuracy:** {accuracy}')
            st.markdown(f'**Model F1 Score:** {f1}')
            st.markdown(
                f'**Predicted Rating for Asset {selected_asset_id}:** {float(predicted_rating):.1f}')

            # Add a horizontal line to separate each nudge
            st.markdown('---')

st.image("pics/apps-logos.png")
