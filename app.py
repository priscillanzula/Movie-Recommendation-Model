import streamlit as st
import pandas as pd
import pickle
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# Load data and models


@st.cache_data
def load_data():
    movies = pd.read_csv('ml-25m/movies.csv')
    ratings = pd.read_csv('ml-25m/ratings.csv')
    return movies, ratings


movies, ratings = load_data()


@st.cache_data
def compute_tfidf_matrix(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))
    return tfidf_matrix


tfidf_matrix = compute_tfidf_matrix(movies)


def content_based_recommendations(movie_id, num_recommendations=5):
    # Get index of the movie
    idx = movies.index[movies['movieId'] == movie_id].tolist()[0]

    # Compute cosine similarities
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get top similar movie indices
    similar_indices = cosine_sim.argsort()[-(num_recommendations+1):][::-1]

    # Exclude the movie itself
    similar_indices = [i for i in similar_indices if i !=
                       idx][:num_recommendations]

    return movies.iloc[similar_indices][['movieId', 'title', 'genres']]


@st.cache_resource
def load_model():
    # In a real deployment, you would load your trained model
    # For demo purposes, we'll train a small model
    sample_frac = 0.01
    ratings_sample = ratings.sample(frac=sample_frac, random_state=42)
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(
        ratings_sample[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    model = SVD(n_factors=50, n_epochs=10, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    return model


model = load_model()

# App layout
st.title('Movie Recommendation System')

# User inputs
st.sidebar.header('User Input')
user_id = st.sidebar.number_input(
    'User ID', min_value=1, max_value=ratings['userId'].max(), value=1)
movie_title = st.sidebar.selectbox('Select a movie you like', movies['title'])

# Get recommendations
if st.sidebar.button('Get Recommendations'):
    movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]

    # Get hybrid recommendations
    st.subheader('Recommended Movies For You')

    # Content-based recommendations
    st.write("### Based on Movie Content")
    content_recs = content_based_recommendations(movie_id, 5)
    st.write(content_recs)

    # Collaborative filtering recommendations
    st.write("### Based on Similar Users")

    # Get top rated movies by similar users
    user_ratings = ratings[ratings['userId'] == user_id]
    if len(user_ratings) > 0:
        # Get movies not rated by user
        rated_movies = user_ratings['movieId'].unique()
        unrated_movies = movies[~movies['movieId'].isin(rated_movies)]

        # Predict ratings for unrated movies
        unrated_movies['predicted_rating'] = unrated_movies['movieId'].apply(
            lambda x: model.predict(user_id, x).est
        )

        # Show top recommendations
        top_collab_recs = unrated_movies.sort_values(
            'predicted_rating', ascending=False).head(5)
        st.write(top_collab_recs[['movieId', 'title', 'genres']])
    else:
        st.write("No user ratings found. Showing popular movies:")
        st.write(movies.sort_values('popularity', ascending=False).head(
            5)[['movieId', 'title', 'genres']])

# Show some statistics
st.sidebar.header('Statistics')
st.sidebar.write(f"Total Movies: {len(movies)}")
st.sidebar.write(f"Total Ratings: {len(ratings)}")
