# Copyright 2020 TalentedSoft ( Author: Shipeng XIA )

import matplotlib.pyplot as plt
import librosa.display
import numpy as py


def load_signal(file_path,sample_rate):
    signal, sample_rate = librosa.load(file_path, sr=sample_rate, mono=True)
    return signal, sample_rate
def VisualWave(title, signal, sample_rate):
        plt.figure(figsize=(8, 4))
        librosa.display.waveplot(signal, sr=sample_rate)
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.savefig('./'+ title + '.jpg')


def load_mel_spectrogram(file_path, n_mels=128, fmax=8000):
    return(librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=n_mels, fmax=fmax))
def VisualSpectrogram(title, spectrogram):
        plt.figure(figsize=(8, 4))
        librosa.display.specshow(
            librosa.power_to_db(spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        plt.colorbar(format='%+10.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()
        plt.savefig('./'+ title + '.jpg')
