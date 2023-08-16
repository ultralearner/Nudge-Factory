# AB Testing.py
# Author: Ganesh Subramanian
# Creation Date: 2023-08-12
# License: MIT
# Description: A Streamlit app to perform A/B testing and visualize the results.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from textblob import TextBlob
import streamlit as st

# Constants
IMAGE_PATH = "pics/"
DATA_PATH = "data/users_synthetic_dataset.csv"

# Suppress warnings
st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


def perform_ab_testing(df_a, df_b):
    # Function to perform A/B Testing including sentiment analysis
    metrics = [
        'Time_Spent', 'Engagement_Score', 'Satisfaction_Score', 'Nudge_Effectiveness',
        'CTR', 'Conversion_Rate', 'Bounce_Rate', 'Retention_Rate', 'Recommendation_Success',
        'Personalization_Effectiveness'
    ]

    # Compare the metrics between the two datasets
    comparison_data = {'Metric': metrics,
                       'Dataset A': [], 'Dataset B': [], 'P-value': []}
    for metric in metrics:
        comparison_data['Dataset A'].append(df_a[metric].mean())
        comparison_data['Dataset B'].append(df_b[metric].mean())

        # Perform an independent t-test
        t_stat, p_val = stats.ttest_ind(df_a[metric], df_b[metric])
        comparison_data['P-value'].append(p_val)

    # Sentiment analysis if Feedback column exists
    if 'Feedback' in df_a.columns and 'Feedback' in df_b.columns:
        sentiment_a = TextBlob(
            df_a['Feedback'].str.cat(sep=' ')).sentiment.polarity
        sentiment_b = TextBlob(
            df_b['Feedback'].str.cat(sep=' ')).sentiment.polarity
        comparison_data['Metric'].append('Sentiment Analysis')
        comparison_data['Dataset A'].append(sentiment_a)
        comparison_data['Dataset B'].append(sentiment_b)
        # No p-value for sentiment analysis
        comparison_data['P-value'].append(None)

    df_comparison = pd.DataFrame(comparison_data)

    return df_comparison


def visualize_ab_testing_results(df_a, df_b):
    # Function to visualize the results of an A/B Testing experiment
    """Visualizes the results of an AB testing experiment.

    Args:
        df_a (pd.DataFrame): The data for dataset A.
        df_b (pd.DataFrame): The data for dataset B.

    Returns:
        None.
    """
    # Combine the two datasets
    df_combined = pd.concat([df_a, df_b], ignore_index=True)

    # Add a column for the user groups label
    df_combined['User Groups'] = df_combined['dataset'].map(
        {'A': 'Group A', 'B': 'Group B'})

    # Confidence Interval Plot for Time_Spent
    st.subheader('Confidence Interval for Time Spent')
    st.markdown('Comparing the Time Spent between User Groups')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='User Groups', y='Time_Spent', data=df_combined, capsize=0.1)
    ax.set_title('Confidence Interval for Time Spent')
    st.pyplot(fig)

    columns_to_compare = ['Time_Spent', 'Engagement_Score',
                          'Nudge_Effectiveness', 'User Groups']

    # Pairwise Comparisons
    st.subheader('Pairwise Comparisons')
    st.markdown('Comparing Key Metrics across User Groups')
    pairplot_fig = sns.pairplot(
        df_combined[columns_to_compare], hue='User Groups')
    st.pyplot(pairplot_fig)

    # Cumulative Distribution Function for Time_Spent
    st.subheader('Cumulative Distribution Function for Time Spent')
    st.markdown('Analyzing Cumulative Probability Distribution')
    fig, ax = plt.subplots()
    for dataset, label in zip([df_a['Time_Spent'], df_b['Time_Spent']], ['Group A', 'Group B']):
        sorted_data = np.sort(dataset)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
        ax.plot(sorted_data, yvals, label=label)
    ax.set_xlabel('Time Spent')
    ax.set_ylabel('Cumulative Probability')
    ax.legend()
    st.pyplot(fig)

    # Violin Plot for Time Spent
    st.subheader('Violin Plot for Time Spent')
    st.markdown('Visualizing Distribution and Probability Density')
    fig, ax = plt.subplots()
    sns.violinplot(x='User Groups', y='Time_Spent', data=df_combined, ax=ax)
    ax.set_title('Violin Plot for Time Spent')
    st.pyplot(fig)

    # 3D Scatter Plot for Time_Spent, Engagement_Score, CTR
    st.subheader('3D Scatter Plot')
    st.markdown(
        'Exploring Relationship between Time Spent, Engagement Score, and CTR')
    fig = plt.figure(figsize=(6, 4))  # You can adjust the size here
    ax = fig.add_subplot(111, projection='3d')

    groups = df_combined['User Groups'].map({'Group A': 'b', 'Group B': 'r'})
    for group, color in zip(['Group A', 'Group B'], ['b', 'r']):
        mask = df_combined['User Groups'] == group
        ax.scatter(df_combined.loc[mask, 'Time_Spent'], df_combined.loc[mask, 'Engagement_Score'],
                   df_combined.loc[mask, 'CTR'], c=color, label=group)

    ax.set_xlabel('Time Spent', fontsize=8)  # Adjust font size here
    ax.set_ylabel('Engagement Score', fontsize=8)  # Adjust font size here
    ax.set_zlabel('CTR', fontsize=8)  # Adjust font size here
    # Adjust X axis tick label font size here
    ax.tick_params(axis='x', labelsize=6)
    # Adjust Y axis tick label font size here
    ax.tick_params(axis='y', labelsize=6)
    # Adjust Z axis tick label font size here
    ax.tick_params(axis='z', labelsize=6)
    ax.legend()  # Add legend
    st.pyplot(fig)


def main():
    # Display the banner
    st.image(f"{IMAGE_PATH}nudgex-banner.png")

    # Read the CSV file
    final_df = load_data(DATA_PATH)

    # Print the summary of the dataset
    st.subheader('Dataset Summary')
    st.write('Number of Rows:', final_df.shape[0])
    st.write('Number of Columns:', final_df.shape[1])
    st.write('Columns:', list(final_df.columns))
    st.write('First 5 Rows:')
    st.dataframe(final_df.head())

    # Filter the datasets
    df_user_activity_a = final_df[final_df['dataset'] == 'A']
    df_user_activity_b = final_df[final_df['dataset'] == 'B']

    # Perform A/B Testing
    df_comparison = perform_ab_testing(
        df_user_activity_a, df_user_activity_b)

    st.subheader('A/B Testing Results')
    st.dataframe(df_comparison)

    # Call the function to visualize the results
    visualize_ab_testing_results(
        df_user_activity_a, df_user_activity_b)

    # Print a divider line
    st.markdown('---')

    # Display the apps logos
    st.image(f"{IMAGE_PATH}apps-logos.png")


if __name__ == "__main__":
    main()
