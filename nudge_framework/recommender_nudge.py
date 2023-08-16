# recommender_nudge.py
# This module is responsible for generating personalized asset recommendations for users.
# Author: Ganesh Subramanian
# Date: 2023-08-12

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
import pickle
from itertools import chain

# Function to load the pre-trained recommender model components


def load_recommender_model():
    with open('models/recommender_model.pkl', 'rb') as f:
        tfidf, tfidf_matrix, cosine_sim, indices, svd = pickle.load(f)
    return tfidf, tfidf_matrix, cosine_sim, indices, svd

# Function to get asset recommendations based on cosine similarity


def get_recommendations(assetName, cosine_sim, asset_df, indices):
    if assetName not in indices:
        return pd.Series([])  # Return an empty series

    idx = indices[assetName]
    sim_scores = sorted(list(
        enumerate(cosine_sim[idx].flatten())), key=lambda x: x[1], reverse=True)[1:6]
    # valid_asset_indices = [i[0] for i in sim_scores if i < len(asset_df)]

    valid_asset_indices = [i[0] for i in sim_scores if i[0] < len(asset_df)]

    return asset_df['assetName'].iloc[valid_asset_indices]

# Function to provide hybrid recommendations by combining content-based and collaborative filtering


def hybrid_recommendations(user_id, ratings_df, asset_df, cosine_sim, svd, indices):
    user_id = int(user_id)
    user_ratings = ratings_df[ratings_df['userId'] == user_id].copy()
    user_ratings['est'] = user_ratings['assetId'].apply(
        lambda x: svd.predict(user_id, x).est)
    top_assets = user_ratings.sort_values(by=['est'], ascending=False)
    top_names = asset_df[asset_df['assetId'].isin(
        top_assets['assetId'].tolist())]['assetName'].tolist()

    similar_assets = [get_recommendations(
        name, cosine_sim, asset_df, indices).tolist() for name in top_names]
    flat_similar_assets = list(chain(*similar_assets))
    recommendations = pd.Series(
        flat_similar_assets, dtype='object').value_counts().index[:5].tolist()

    # Fill with top rated assets if not enough recommendations
    if len(recommendations) < 5:
        top_overall = ratings_df['assetId'].value_counts().index[:5].tolist()
        for asset_id in top_overall:
            if len(recommendations) == 5:
                break
            asset_name = asset_df[asset_df['assetId']
                                  == asset_id]['assetName'].values[0]
            if asset_name not in recommendations:
                recommendations.append(asset_name)

    return recommendations

# Function to generate final recommendations


def generate_recommendations(user_id, tfidf, tfidf_matrix, cosine_sim, indices, svd):
    # Load data
    asset_df = pd.read_csv('data/asset_catalog.csv')
    ratings_df = pd.read_csv(
        'data/master_dataset.csv')

    # Generate the recommendations for the user
    # Pass indices as an argument
    return hybrid_recommendations(user_id, ratings_df, asset_df, cosine_sim, svd, indices)


def test_generate_recommendations_existing_user():
    tfidf, tfidf_matrix, cosine_sim, indices, svd = load_recommender_model()
    recommendations = generate_recommendations(
        3001, tfidf, tfidf_matrix, cosine_sim, indices, svd)
    assert isinstance(recommendations, list), "The output should be a list."
    assert len(
        recommendations) == 5, "The output list should contain 5 recommendations."
    print("Test Case 1 Passed")


test_generate_recommendations_existing_user()


# def test_generate_recommendations_non_existing_user():
#     tfidf, tfidf_matrix, cosine_sim, indices, svd = load_recommender_model()
#     recommendations = generate_recommendations(
#         999999, tfidf, tfidf_matrix, cosine_sim, indices, svd)
#     assert isinstance(recommendations, list), "The output should be a list."
#     assert len(
#         recommendations) == 5, "The output list should contain 5 recommendations."
#     print("Test Case 2 Passed")


# test_generate_recommendations_non_existing_user()
