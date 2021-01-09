import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#from tensorflow_asr.augmentations.augments import Augmentation
from  tensorflow_asr.featurizers.speech_featurizers import read_raw_audio
from tensorflow_asr.configs.config import Config

import librosa

import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import soundfile



def visual(title, audio, sample_rate):
        plt.figure(figsize=(8, 4))
        librosa.display.waveplot(audio, sr=sample_rate)
        plt.title(title)
        plt.tight_layout()
        #plt.show()
        plt.savefig("./audio_b.jpg")

config_dir = "tests/config_aishell.yml"              
config = Config(config_dir, learning=True)


aug=config.learning_config.augmentations

sampling_rate=16000
audio = '/tsdata/ASR/aishell-1//wav/train/S0002/BAC009S0002W0123.wav'
signal = read_raw_audio(audio, sampling_rate)

#visual('Original', signal, sampling_rate)

signal = aug.before.augment(signal)
visual('Original', signal, sampling_rate)
soundfile.write('./test_d.wav',signal,16000)