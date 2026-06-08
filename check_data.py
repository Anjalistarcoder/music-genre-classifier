import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")

print("X Shape:", X.shape)
print("y Shape:", y.shape)

print("\nUnique Genres:")
print(np.unique(y))

print("\nGenre Counts:")
unique, counts = np.unique(y, return_counts=True)

for g, c in zip(unique, counts):
    print(g, ":", c)