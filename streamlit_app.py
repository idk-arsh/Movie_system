import streamlit as st
import pandas as pd
from recommendation import UnifiedRecommendationSystem  # Import your class

@st.cache_data
def load_data():
    return pd.read_excel('./dataset.xlsx')

data = load_data()


best_params = {'n_factors': 100, 'reg_all': 0.05}


@st.cache_resource
def initialize_model(data, params):
    rec_system = UnifiedRecommendationSystem(data, params)
    rec_system.train_model()
    return rec_system


rec_system = initialize_model(data, best_params)

st.title("Movie Recommendation System")

st.write("Select Your Preferred Genres:")

genres = {
    "Action": "Action",
    "Adventure": "Adventure",
    "Animation": "Animation",
    "Children's": "Children",
    "Comedy": "Comedy",
    "Crime": "Crime",
    "Documentary": "Documentary",
    "Drama": "Drama",
    "Fantasy": "Fantasy",
    "Horror": "Horror",
    "Musical": "Musical",
    "Mystery": "Mystery",
    "Romance": "Romance",
    "Sci-Fi": "SciFi",
    "Thriller": "Thriller",
    "War": "War",
    "Western": "Western"
}

selected_genres = []
for genre_name, genre_value in genres.items():
    if st.checkbox(genre_name):
        selected_genres.append(genre_value)

# Fetch recommendations
if st.button("Get Recommendations"):
    if selected_genres:
        recommendations = rec_system.get_recommendations(preferences={'genres': selected_genres})
    else:
        recommendations = rec_system.get_recommendations()

    st.write("Recommended Movies:")
    for movie in recommendations:
        st.write(f"- {movie}")
