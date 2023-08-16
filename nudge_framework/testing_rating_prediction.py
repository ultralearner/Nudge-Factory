import random
from user_rating_prediction import predict_rating

# Function to test predictions for 100 users and 25 assets


def test_predictions():
    user_ids = [random.randint(3000, 3300) for _ in range(100)]
    asset_ids = [random.randint(1, 311) for _ in range(25)]

    predictions = []

    for user_id in user_ids:
        user_predictions = []
        for asset_id in asset_ids:
            predicted_rating = predict_rating(user_id, asset_id)
            user_predictions.append(predicted_rating)
            print(
                f"Predicted rating for User {user_id} and Asset {asset_id}: {predicted_rating}")
        predictions.append(user_predictions)

    return predictions


# Call the test function
test_predictions()
