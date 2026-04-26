import numpy as np
import librosa


def aggregate_feature(feature_matrix):
    mean = np.mean(feature_matrix, axis=1)
    std = np.std(feature_matrix, axis=1)
    return np.concatenate([mean, std])


def extract_features_from_audio(audio, sr, feature_set="basic"):
    features = []

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512,
        window="hann"
    )

    features.append(aggregate_feature(mfcc))

    if feature_set == "basic":
        return np.concatenate(features)

    raise ValueError("Tato aplikace aktuálně používá pouze basic sadu příznaků.")