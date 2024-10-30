import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestion
from src.components.recommender import Recommender
from src.exception import CustomException
from src.logger import logging
from src.utils import pick_random_song

# Configuration dataclass for data ingestion
@dataclass
class RecommendPipeline:
    def __init__(self):
        pass

    def recommend(self, song_name:str):
        try:
            # Step 1: Data Ingestion
            ingestion = DataIngestion()
            data_path = ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            data_transformation = DataTransformation()
            df, tfidf_matrix = data_transformation.initiate_data_transformation(data_path)

            # Step 3: Recommender Initialization and Recommendation Generation
            recommender = Recommender(
                df=df,  # Pass the processed DataFrame
                tfidf_matrix=tfidf_matrix  # Pass the TF-IDF matrix
            )

            # Get a random song and generate recommendations
            logging.info(f"Selected random song for recommendation: {song_name}")

            recommendations = recommender.recommend(song_name=song_name)
            return recommendations

        except Exception as e:
            raise CustomException(e, sys)

# Run the Recommendation Pipeline
if __name__ == "__main__":
    recommend_pipeline = RecommendPipeline()
    recommendations = recommend_pipeline.recommend()
    print("Recommended Songs:", recommendations)