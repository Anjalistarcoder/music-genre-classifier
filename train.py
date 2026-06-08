import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint

# ==========================
# Load Dataset
# ==========================

X = np.load("X.npy")
y = np.load("y.npy")

print("Original X Shape:", X.shape)
print("Original y Shape:", y.shape)

# Normalize data
X = X.astype("float32")

# Convert -80..0 range to 0..1 range
X = (X + 80.0) / 80.0

# Add channel dimension
X = X[..., np.newaxis]

# ==========================
# Encode Labels
# ==========================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

# ==========================
# CNN Model
# ==========================

model = models.Sequential()

model.add(
    layers.Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=X_train.shape[1:]
    )
)

model.add(layers.MaxPooling2D((2,2)))

model.add(
    layers.Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(layers.MaxPooling2D((2,2)))

model.add(
    layers.Conv2D(
        128,
        (3,3),
        activation="relu"
    )
)

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(
    layers.Dense(
        128,
        activation="relu"
    )
)

model.add(layers.Dropout(0.3))

model.add(
    layers.Dense(
        10,
        activation="softmax"
    )
)



# ==========================
# Compile Model
# ==========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================
# Callbacks
# ==========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_music_genre_model.keras",
    save_best_only=True
)

# ==========================
# Train Model
# ==========================

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
)

# ==========================
# Evaluate
# ==========================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nFinal Accuracy:", accuracy)

# ==========================
# Save Final Model
# ==========================

model.save("music_genre_model.keras")

print("Model Saved Successfully!")