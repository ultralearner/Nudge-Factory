import os
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import pickle


def train_and_save_prediction_model():
    # Load the data from CSV file into pandas DataFrame
    df = pd.read_csv(
        'data/master_dataset.csv')

    # Initialize a reader to read the data
    reader = Reader(rating_scale=(1, 5))

    # Load the data
    data = Dataset.load_from_df(df[['userId', 'assetId', 'rating']], reader)

    # Initialize the SVD algorithm
    algo = SVD()

    # Cross-validate the algorithm on the data
    cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Fit the algorithm to the entire dataset
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    # Make sure the models directory exists
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the trained model to a file
    with open(os.path.join(model_dir, 'rating_prediction_model.pkl'), 'wb') as f:
        pickle.dump(algo, f)


def load_rating_prediction_model():
    model_path = 'models/rating_prediction_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        # print("Model loaded successfully.")
    else:
        print("Model file not found. Please check the path or train the model.")
        model = None  # or could return some default model here
    return model


def predict_rating(user_id, asset_id):
    # Load the model
    model = load_rating_prediction_model()

    # Use the model to predict the rating
    predicted = model.predict(int(user_id), str(asset_id))

    # Return the estimated rating
    return predicted.est
