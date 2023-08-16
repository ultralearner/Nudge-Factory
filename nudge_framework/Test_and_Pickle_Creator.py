import joblib
from onboarding_nudge import generate_onboarding_nudge
from engagement_nudge import generate_engagement_nudge
from feedback_nudge import generate_feedback_nudge
from train_model_recommender import train_model_recommender
from user_rating_prediction import train_and_save_prediction_model

# Generate the onboarding nudge for a user and print the results
nudge, explanation, accuracy, f1 = generate_onboarding_nudge(3000)
print("Nudge: ", nudge)
print("Explanation: ", explanation)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)
print("Onboarding nudge generated successfully.")

# Generate the engagement nudge for a user and print the results
user_ids = [3001, 3100, 3150, 4000]
trigger = "Long User Inactivity"  # replace with actual trigger
for user_id in user_ids:
    nudge, explanation, accuracy, f1 = generate_engagement_nudge(
        user_id, trigger)
    print(f"Nudge: {nudge}")
    print(f"Explanation: {explanation}")
    print(f"Model Accuracy: {accuracy}")
    print(f"Model F1 Score: {f1}")
print("Engagement nudges generated successfully.")

# Generate the feedback nudge for a user and print the results
user_id = 3283  # Test user ID
trigger = "Negative Feedback"
nudge, explanation, accuracy, f1 = generate_feedback_nudge(user_id, trigger)

print(f"Nudge: {nudge}")
print(f"Explanation: {explanation}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print("Feedback nudge generated successfully.")


# Train the recommender model to create the pickle file
train_model_recommender()
print("Recommender model trained successfully.")

# Train the prediction model to create the pickle file
train_and_save_prediction_model()
print("Prediction model trained successfully.")
