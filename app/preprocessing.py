import numpy as np
import librosa


def aggregate_feature(feature_matrix):
    mean = np.mean(feature_matrix, axis=1)
    std = np.std(feature_matrix, axis=1)
    return np.concatenate([mean, std])


def extract_features_from_audio(audio, sr):
    features = []

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512,
        window="hann"
    )

    for i in range(mfcc.shape[0]):
        features.append(np.mean(mfcc[i]))
        features.append(np.std(mfcc[i]))

    return np.array(features)