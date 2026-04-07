import numpy as np
import librosa


def aggregate_feature(feature_matrix):
    mean = np.mean(feature_matrix, axis=1)
    std = np.std(feature_matrix, axis=1)
    return np.concatenate([mean, std])


def extract_features_from_audio(audio, sr, feature_set="advanced"):
    features = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features.append(aggregate_feature(mfcc))

    if feature_set in ["extended", "advanced"]:
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(aggregate_feature(zcr))

        rms = librosa.feature.rms(y=audio)
        features.append(aggregate_feature(rms))

        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(aggregate_feature(centroid))

        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.append(aggregate_feature(bandwidth))

        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(aggregate_feature(rolloff))

        flatness = librosa.feature.spectral_flatness(y=audio)
        features.append(aggregate_feature(flatness))

    if feature_set == "advanced":
        delta_mfcc = librosa.feature.delta(mfcc)
        features.append(aggregate_feature(delta_mfcc))

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.append(aggregate_feature(chroma))

    return np.concatenate(features)