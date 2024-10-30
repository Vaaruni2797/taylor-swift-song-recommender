import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.recommender import Recommender
from src.components.recommender import RecommenderConfig

from src.exception import CustomException
from src.logger import logging
from src.utils import pick_random_song



@dataclass
class DataIngenstionConfig:
    data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngenstionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("data/taylor_swift_tracks_dataset.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.data_path, index =False, header = True)

            logging.info("Ingestion Completed")

            return (
                self.ingestion_config.data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=="__main__":
    obj = DataIngestion()
    data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    df,tfidf_matrix = data_transformation.initiate_data_transformation(data_path)

    # Recommender Initialization
    recommender = Recommender(
        df=df,  # Pass the processed DataFrame
        tfidf_matrix=tfidf_matrix  # Pass the TF-IDF matrix
    )
    song_name = pick_random_song(df)

    # Example: Recommend based on a song name
    recommendations = recommender.recommend(song_name=song_name)  # Provide a valid song name from your dataset
    print(recommendations)