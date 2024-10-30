import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from src.exception import CustomException
from src.logger import logging

class DataTransformation:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.sid = SentimentIntensityAnalyzer()

    def preprocess_lyrics(self, lyrics):
        '''
        Function to preprocess lyrics (tokenization + lemmatization)
        '''
        try:
            tokens = word_tokenize(lyrics)
            lemmatized_tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha()]
            return ' '.join(lemmatized_tokens)
        except Exception as e:
            raise CustomException(e, sys)

    def get_sentiment(self, lyrics):
        '''
        Function to calculate sentiment score for each song's lyrics
        '''
        try:
            if lyrics:
                sentiment = self.sid.polarity_scores(lyrics)
                return sentiment['compound']  # Compound sentiment score (-1 to +1)
            return 0  # Neutral if no lyrics available
        except Exception as e:
            raise CustomException(e, sys)

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

    def initiate_data_transformation(self, data_path):
        try:
            df = pd.read_csv(data_path)
            # Handle missing data
            df['lyrics'] = df['lyrics'].fillna('')
            df['mood'] = df['mood'].fillna('neutral')

            # Select audio features to include in similarity calculation
            audio_features = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']

            # Normalize audio features
            scaler = MinMaxScaler()
            df[audio_features] = scaler.fit_transform(df[audio_features])
            logging.info("Audio features normalization completed")

            # Apply preprocessing to lyrics
            df['processed_lyrics'] = df['lyrics'].apply(self.preprocess_lyrics)
            logging.info("Lyrics preprocessing completed")

            # Initialize TF-IDF Vectorizer for lyrics
            tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_lyrics'])
            logging.info("TF-IDF vectorization of lyrics completed")

            # Apply sentiment analysis to lyrics
            df['sentiment'] = df['lyrics'].apply(self.get_sentiment)
            logging.info("Sentiment analysis completed")

            return df, tfidf_matrix  # Return the transformed dataframe and TF-IDF matrix

        except Exception as e:
            raise CustomException(e, sys)
