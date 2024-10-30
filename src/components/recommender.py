import os
import sys
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from src.exception import CustomException
from src.logger import logging

@dataclass
class RecommenderConfig:
    num_recommendations: int = 5
    mood_weight: float = 0.3
    audio_weight: float = 0.4

class Recommender:
    def __init__(self, df, tfidf_matrix):
        self.df = df
        self.tfidf_matrix = tfidf_matrix
        self.recommender_config = RecommenderConfig()

    def extract_base_title(self, title):
        '''
        Helper function to extract base title from the full title
        '''
        try:
            base_title = title.lower().split('(')[0].strip()
            return base_title
        except Exception as e:
            raise CustomException(e, sys)

    def mood_similarity(self, mood1, mood2):
        '''
        Helper function to calculate mood similarity
        '''
        return 1 if mood1 == mood2 else 0

    def get_sentiment_difference(self, song1_sentiment, song2_sentiment):
        '''
        Calculate sentiment difference between two songs
        '''
        return np.abs(song1_sentiment - song2_sentiment)

    def recommend(self, song_name):
        '''
        Function to recommend similar songs based on lyrics and sentiment
        '''
        try:
            # Find the index of the song in the dataframe
            idx = self.df[self.df['track_name'].str.lower() == song_name.lower()].index[0]
            
            # Get the mood and audio features of the input song
            input_song_mood = self.df.iloc[idx]['mood']

            
            audio_features = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']
            input_song_audio_features = self.df.iloc[idx][audio_features].values.reshape(1, -1)
            
            # Compute cosine similarity for lyrics
            cosine_similarities_lyrics = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
            
            # Compute cosine similarity for audio features
            cosine_similarities_audio = cosine_similarity(input_song_audio_features, self.df[audio_features]).flatten()

            # Extract sentiment of the input song
            input_song_sentiment = self.df.iloc[idx]['sentiment']
            
            # Extract the base title of the input song to exclude its versions
            base_title_input = self.extract_base_title(song_name)
            
            # Get indices of the top similar songs, excluding the input song and its versions
            similar_indices = cosine_similarities_lyrics.argsort()[::-1]
            
            # Filter out the input song's versions and duplicate base titles
            recommended_songs = []
            for i in similar_indices:
                if self.df.iloc[i]['track_name'].lower() == song_name.lower():
                    continue  # Skip the input song itself
                
                base_title_recommendation = self.extract_base_title(self.df.iloc[i]['track_name'])
                
                # Exclude songs that have the same base title as the input song or previously recommended songs
                if base_title_recommendation != base_title_input and base_title_recommendation not in [self.extract_base_title(song['track_name']) for song, _ in recommended_songs]:
                    # Calculate mood similarity
                    mood_sim = self.mood_similarity(input_song_mood, self.df.iloc[i]['mood'])
                    
                    # Combine lyrics similarity, mood similarity, and audio similarity using weights
                    combined_score = ((1 - self.recommender_config.mood_weight - self.recommender_config.audio_weight) * cosine_similarities_lyrics[i] +
                                      self.recommender_config.mood_weight * mood_sim +
                                      self.recommender_config.audio_weight * cosine_similarities_audio[i])
                    
                    # Calculate sentiment difference
                    sentiment_difference = self.get_sentiment_difference(self.df.iloc[i]['sentiment'], input_song_sentiment)
                    
                    # Only recommend songs with a similar sentiment (difference below a certain threshold)
                    if sentiment_difference < 0.2: 
                        recommended_songs.append((self.df.iloc[i], combined_score))
                
                # Stop when we have enough unique recommendations
                if len(recommended_songs) == self.recommender_config.num_recommendations:
                    break
            
            # Sort the recommendations by combined score
            recommended_songs = sorted(recommended_songs, key=lambda x: x[1], reverse=True)
            
            # Prepare and return the list of recommendations
            recommended_songs_list = [
                {
                    "track_name": song['track_name'],
                    "album_name": song['album_name'],
                    "score": score,
                    "mood": song['mood'],
                    "sentiment": song['sentiment']
                }
                for song, score in recommended_songs
            ]
            
            return recommended_songs_list

        except Exception as e:
            raise CustomException(e, sys)