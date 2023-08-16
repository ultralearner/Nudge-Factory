import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
import pickle
import os

# move the code to a separate function


def train_model_recommender():

    # Load data
    asset_df = pd.read_csv('data/asset_catalog.csv')
    ratings_df = pd.read_csv(
        'data/master_dataset.csv')

    # Preprocess text data
    asset_df['assetName'] = asset_df['assetName'].fillna('')
    asset_df['assetDesc'] = asset_df['assetDesc'].fillna('')
    asset_df['text'] = asset_df['assetName'] + ' ' + asset_df['assetDesc']

    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(asset_df['text'])

    # Calculate cosine similarities
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Create mapping from asset names to indices
    indices = pd.Series(
        asset_df.index, index=asset_df['assetName']).drop_duplicates()

    # Train SVD model
    reader = Reader()
    data = Dataset.load_from_df(
        ratings_df[['userId', 'assetId', 'rating']], reader)
    svd = SVD()
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    model_dir = 'models'
    model_filepath = os.path.join(model_dir, 'recommender_model.pkl')

    # Create the directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the models
    with open(model_filepath, 'wb') as f:
        pickle.dump((tfidf, tfidf_matrix, cosine_sim, indices, svd), f)

    # Validate that the model file was created
    if os.path.exists(model_filepath):
        print(f"Model saved successfully at {model_filepath}")
    else:
        print(f"Failed to save model at {model_filepath}")
