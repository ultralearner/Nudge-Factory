# Filename: Home.py
# Author: Ganesh Subramanian
# Date Created: 2023-08-12
# License: MIT
# Description: Home page for the NudgeX platform, displaying welcome message, about section, and application logos.

import streamlit as st

# Path to images directory
image_path = "pics/"

# Display the banner
st.image(image_path + "nudgex-banner.png")

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
st.image(image_path + "apps-logos.png")
