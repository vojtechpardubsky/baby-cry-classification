import numpy as np
import librosa

N_MFCC = 13
WIN_LENGTH = 256
N_FFT = 512
HOP_LENGTH = 80

def aggregate_feature(feature_matrix):
    features = []
    for i in range(feature_matrix.shape[0]):
        values = feature_matrix[i]
        features.append(np.mean(values))
        features.append(np.std(values))
    return np.array(features)


def extract_features_from_audio(audio, sr):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann"
    )

    return aggregate_feature(mfcc)