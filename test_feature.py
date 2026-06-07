from feature_extraction import extract_features

file_path = "dataset/genres_original/blues/blues.00000.wav"

features = extract_features(file_path)

print("Feature Shape:", features.shape)