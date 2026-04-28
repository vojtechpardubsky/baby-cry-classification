import numpy as np
import librosa


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
        n_mfcc=13,
        n_fft=2048,
        hop_length=512,
        window="hann"
    )

    return aggregate_feature(mfcc)