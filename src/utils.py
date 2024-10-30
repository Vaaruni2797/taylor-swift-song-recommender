import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def pick_random_song(data):
    random_index = random.randint(0, len(data) - 1)  # Pick a random index
    random_song = data.iloc[random_index]['track_name']
    print(f"\nRandomly picked song: {random_song}")
    return random_song