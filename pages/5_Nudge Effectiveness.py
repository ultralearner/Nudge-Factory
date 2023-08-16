# Nudge Effectiveness.py
# Author: Ganesh Subramanian
# Creation Date: 2023-08-12
# License: MIT
# Description: A Streamlit app for analyzing and visualizing the effectiveness of nudges.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Constants
IMAGE_PATH = "pics/"
DATA_PATH = "data/users_synthetic_dataset.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def main():
    # Display the banner
    st.image(f"{IMAGE_PATH}nudgex-banner.png")

    data = load_data()

    # Convert 'Timestamp' to datetime format
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Minute'] = data['Timestamp'].dt.floor('T')  # Resampling by minute

    # Title
    st.title("Nudge Effectiveness Analysis")

    # Summary Statistics
    st.subheader("Summary Statistics")
    nudge_effectiveness_summary = data.groupby(
        'Category')['Nudge_Effectiveness'].agg(['mean', 'median', 'std'])
    st.write(nudge_effectiveness_summary)

    # Boxplot
    st.subheader("Distribution of Nudge Effectiveness")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Category', y='Nudge_Effectiveness', data=data)
    plt.title('Distribution of Nudge Effectiveness across Categories')
    plt.xlabel('Category')
    plt.ylabel('Nudge Effectiveness')
    st.pyplot(plt.gcf())

    # Nudge Effectiveness Over Time (Minute-by-Minute)
    st.subheader("Nudge Effectiveness Over Time (Minute-by-Minute)")
    time_trend_minute = data.groupby(['Minute', 'Category'])[
        'Nudge_Effectiveness'].mean().reset_index()
    plt.figure(figsize=(15, 6))
    sns.lineplot(x='Minute', y='Nudge_Effectiveness',
                 hue='Category', data=time_trend_minute)
    plt.title('Trend of Nudge Effectiveness by Minute')
    plt.xlabel('Minute')
    plt.ylabel('Nudge Effectiveness')
    # Optional: Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

    # Print a divider line
    st.markdown('---')

    # Display the apps logos
    st.image(f"{IMAGE_PATH}apps-logos.png")


if __name__ == "__main__":
    main()
