import pandas as pd
import numpy as np
import random
from datetime import datetime
from textblob import TextBlob

from faker import Faker
from textblob import TextBlob

fake = Faker()

# Function to generate synthetic data


def generate_synthetic_data(n_interactions, dataset_label):
    data = []
    sentiments = ["Positive", "Negative", "Neutral"]

    # Triggers for each category
    triggers = {
        "Onboarding": ["Signup-Login"],
        "Engagement": ["Long User Inactivity", "Near Top Contributor", "New Assets Added", "Used, Not Rated", "Used, No Feedback"],
        "Feedback": ["Frequent Portal Usage", "Negative Feedback Given", "New Asset, No Feedback", "Used An Asset", "Used New Asset"]
    }

    for i in range(n_interactions):
        user_id = random.randint(3000, 3300)
        asset_id = random.randint(1, 311)
        timestamp = pd.Timestamp.now() + pd.Timedelta(seconds=i)
        time_spent = random.uniform(10, 300)  # Time spent in seconds
        engagement_score = random.randint(1, 100)
        satisfaction_score = random.randint(1, 100)
        ctr = random.uniform(0, 1)
        conversion_rate = random.uniform(0, 1)
        bounce_rate = random.uniform(0, 1)
        retention_rate = random.uniform(0, 1)
        recommendation_success = random.uniform(0, 1)
        personalization_effectiveness = random.uniform(0, 1)
        nudge_effectiveness = (
            ctr * conversion_rate * engagement_score + satisfaction_score - bounce_rate) / 100

        # Selecting a random category and trigger
        category = random.choice(list(triggers.keys()))
        trigger = random.choice(triggers[category])

        feedback_text = fake.sentence()
        sentiment = TextBlob(feedback_text).sentiment.polarity
        sentiment_label = sentiments[0 if sentiment >
                                     0 else 1 if sentiment < 0 else 2]

        data.append([
            user_id, asset_id, timestamp, time_spent, engagement_score, satisfaction_score,
            nudge_effectiveness, ctr, conversion_rate, bounce_rate, retention_rate,
            recommendation_success, personalization_effectiveness, sentiment_label,
            category, trigger  # Adding the new columns
        ])

    df = pd.DataFrame(data, columns=[
        'User_ID', 'Asset_ID', 'Timestamp', 'Time_Spent', 'Engagement_Score', 'Satisfaction_Score',
        'Nudge_Effectiveness', 'CTR', 'Conversion_Rate', 'Bounce_Rate', 'Retention_Rate',
        'Recommendation_Success', 'Personalization_Effectiveness', 'Sentiment Label',
        'Category', 'Trigger'  # Including the new columns in DataFrame
    ])

    df['dataset'] = dataset_label

    return df


n_interactions = 1000
dataset_label_a = 'A'
dataset_label_b = 'B'
df_a = generate_synthetic_data(n_interactions, dataset_label_a)
df_b = generate_synthetic_data(n_interactions, dataset_label_b)

# Combine both DataFrames
final_df = pd.concat([df_a, df_b], ignore_index=True)

# Shuffle the DataFrame to randomly mix 'A' and 'B' dataset labels
final_df = final_df.sample(frac=1).reset_index(drop=True)

# Save to CSV
final_df.to_csv('data/users_synthetic_dataset.csv', index=False)
