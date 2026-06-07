import librosa
import numpy as np

def extract_features(file_path, max_pad_len=128):

    try:
        audio, sr = librosa.load(file_path, sr=22050)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=128
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        if mel_db.shape[1] < max_pad_len:

            pad_width = max_pad_len - mel_db.shape[1]

            mel_db = np.pad(
                mel_db,
                pad_width=((0,0),(0,pad_width)),
                mode='constant'
            )

        else:
            mel_db = mel_db[:, :max_pad_len]

        return mel_db

    except Exception as e:
        print(e)
        return None