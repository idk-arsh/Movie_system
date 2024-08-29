import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise import accuracy
import re

class UnifiedRecommendationSystem:
    def __init__(self, data, best_params):
        self.data = data
        self.best_params = best_params
        self.model = None
        self.preprocessed_data = self.preprocessing_data()

    def preprocessing_data(self):
        # Create dummy variables for genres
        df_genres = self.data['genres'].str.get_dummies('|')
        self.data = pd.concat([self.data, df_genres], axis=1)
        # Add genre count and release year columns
        self.data['genres_count'] = self.data['genres'].apply(lambda x: len(x.split('|')))
        self.data['release_year'] = self.data['title'].apply(
            lambda x: int(re.search(r'\((\d{4})\)', x).group(1)) if re.search(r'\((\d{4})\)', x) else None
        )
        return self.data

    def train_model(self):
        # Prepare the data for the model
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.data[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        # Initialize and train the SVD model
        self.model = SVD(n_factors=self.best_params['n_factors'], reg_all=self.best_params['reg_all'])
        self.model.fit(trainset)

    def evaluate_model(self):
        testset = self.model.trainset.build_testset()
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions)
        return rmse

    def predict(self, user_id, movie_id):
        prediction = self.model.predict(user_id, movie_id)
        return prediction.est

    def get_recommendations(self, user_id=None, preferences=None):
        if user_id is not None and user_id in self.data['userId'].unique():
            return self.get_recommendations_for_existing_user(user_id)
        else:
            return self.get_recommendations_for_new_user(preferences)

    def get_recommendations_for_existing_user(self, user_id):
        # Get the list of items the user has not rated yet
        user_items = set(self.data[self.data['userId'] == user_id]['movieId'])
        all_items = set(self.data['movieId'])
        unrated_items = list(all_items - user_items)
        # Predict ratings for unrated items
        predictions = [(item, self.predict(user_id, item)) for item in unrated_items]
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]
        # Get movie titles for recommended items
        recommended_movies = [self.data[self.data['movieId'] == movie_id]['title'].iloc[0] for movie_id in recommendations]
        return recommended_movies

    def get_recommendations_for_new_user(self, preferences=None):
        if preferences:
            # Convert selected genres to the appropriate format for filtering
            preferred_genres = [f'{genre}' for genre in preferences.get('genres', [])]
            genre_mask = self.preprocessed_data[preferred_genres].sum(axis=1) > 0
            genre_df = self.preprocessed_data[genre_mask]
            if not genre_df.empty:
                # Recommend top-rated movies within selected genres
                recommendations = genre_df.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(20).index.tolist()
            else:
                # Fall back to popular items if no movies match the selected genres
                recommendations = self.get_popular_items()
        else:
            recommendations = self.get_popular_items()

        # Get movie titles for recommended items
        recommended_movies = [self.data[self.data['movieId'] == movie_id]['title'].iloc[0] for movie_id in recommendations]
        return recommended_movies

    def get_popular_items(self, top_n=20):
        # Recommend the most popular items based on rating counts
        popularity = self.data.groupby('movieId')['rating'].count().sort_values(ascending=False)
        popular_items = popularity.index[:top_n].tolist()
        return popular_items
