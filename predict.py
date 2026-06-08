import numpy as np
from keras.models import load_model
from feature_extraction import extract_features

# Load model
model = load_model("music_genre_model.keras")

# Genre names
genres = [
    'blues',
    'classical',
    'country',
    'disco',
    'hiphop',
    'jazz',
    'metal',
    'pop',
    'reggae',
    'rock'
]

# Test audio file
file_path = "dataset/genres_original/rock/rock.00000.wav"

# Extract features
features = extract_features(file_path)

features = np.expand_dims(features, axis=0)
features = np.expand_dims(features, axis=-1)

prediction = model.predict(features)

predicted_genre = genres[np.argmax(prediction)]

print("Predicted Genre:", predicted_genre)