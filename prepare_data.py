import os
import numpy as np
from tqdm import tqdm

from feature_extraction import extract_features

X = []
y = []

dataset_path = "dataset/genres_original"

genres = os.listdir(dataset_path)

print("Genres Found:")
print(genres)

for genre in tqdm(genres):

    genre_path = os.path.join(dataset_path, genre)

    for file in os.listdir(genre_path):

        file_path = os.path.join(genre_path, file)

        features = extract_features(file_path)

        if features is not None:

            X.append(features)
            y.append(genre)

X = np.array(X)

np.save("X.npy", X)
np.save("y.npy", np.array(y))

print("\nDataset Saved Successfully!")
print("X Shape:", X.shape)
print("Total Labels:", len(y))