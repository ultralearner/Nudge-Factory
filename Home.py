import os
import streamlit as st

# Function to display images if they exist


def display_image(image_name):
    image_path = os.path.join("pics", image_name)
    if os.path.exists(image_path):
        st.image(image_path)
    else:
        st.warning(f"Image {image_name} not found!")


# Display the banner
display_image("nudgex-banner.png")

# Display the welcome message
st.write("""
# Welcome to NudgeX!
NudgeX is a cutting-edge AI-driven platform that uses nudges to enhance user engagement and productivity.
""")

# Display the About NudgeX section
st.write("""
## About NudgeX
NudgeX leverages behavioral science and machine learning to encourage users to interact with the platform in a more engaging and productive manner. It employs a range of nudges, each designed to cater to different user behaviors and contexts. The nudges are dynamically adjusted based on real-time user interaction data, ensuring that the right nudge is delivered to the right user at the right time. This results in a personalized and seamless user experience that enhances user satisfaction and productivity.
""")

# Display the apps logos
display_image("apps-logos.png")
