import numpy as np

X = np.load("X.npy")

print("Shape:", X.shape)
print("Min:", np.min(X))
print("Max:", np.max(X))
print("Mean:", np.mean(X))
print("Contains NaN:", np.isnan(X).any())
print("Contains Inf:", np.isinf(X).any())