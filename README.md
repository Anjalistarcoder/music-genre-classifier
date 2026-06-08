# 🎵 Music Genre Classification using CNN

A Deep Learning project that classifies music into different genres using Mel Spectrograms and a Convolutional Neural Network (CNN).

## 📌 Project Overview

This project uses the GTZAN Music Genre Dataset and Deep Learning techniques to automatically classify songs into one of 10 music genres.

The audio files are converted into Mel Spectrograms using Librosa, and a CNN model is trained using TensorFlow/Keras for genre prediction.

## 🎯 Genres Classified

* Blues
* Classical
* Country
* Disco
* Hip-Hop
* Jazz
* Metal
* Pop
* Reggae
* Rock

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Librosa
* NumPy
* Scikit-Learn
* Matplotlib

---

## 📂 Project Structure

```text
music-genre-classifier/
│
├── feature_extraction.py
├── prepare_data.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── sample_audio/
```

---

## 📊 Dataset

Dataset Used: GTZAN Genre Collection

* 1000 audio clips
* 10 genres
* 30-second audio samples

---

## ⚙️ Feature Extraction

Mel Spectrograms are extracted from audio files using Librosa.

Features Generated:

* 128 Mel Bands
* Spectrogram Size: 128 × 128
* Shape: (999, 128, 128)

---

## 🧠 Model Architecture

CNN Architecture:

```text
Conv2D (32 filters)
↓
MaxPooling2D
↓
Conv2D (64 filters)
↓
MaxPooling2D
↓
Conv2D (128 filters)
↓
MaxPooling2D
↓
Flatten
↓
Dense (128)
↓
Dropout (0.3)
↓
Dense (10 Softmax)
```

---

## 🚀 Training

Run:

```bash
python prepare_data.py
```

Then:

```bash
python train.py
```

The trained model will be saved as:

```text
music_genre_model.keras
```

---

## 🎵 Prediction

To predict the genre of a new audio file:

```bash
python predict.py
```

Example Output:

```text
Predicted Genre: Classical
```

---

## 📈 Results

| Metric              | Value       |
| ------------------- | ----------- |
| Dataset Size        | 999 Samples |
| Number of Genres    | 10          |
| Spectrogram Shape   | 128 × 128   |
| Final Test Accuracy | 52.5%       |

---

## 💡 Future Improvements

* Audio Segmentation (3-second chunks)
* Data Augmentation
* Transfer Learning
* Improved CNN Architectures
* Real-time Genre Prediction

---

## 👩‍💻 Author

Anjali Singh

GitHub: https://github.com/Anjalistarcoder

LinkedIn: https://www.linkedin.com/in/anjali-singh-884bb0375/

---

## ⭐ If you found this project useful, please give it a star!
