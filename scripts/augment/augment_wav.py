# Copyright 2020 TalentedSoft ( Author: Shipeng XIA )

import os
import soundfile
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow_asr.configs.config import Config
from scripts.visual import load_signal


sample_rate=16000
config_dir = "scripts/augment/config_augment.yml"  
file_path="/work/kaldi/egs/XSP/TensorFlowASR/data/Aishell_1/test_transcripts.tsv"
output_path="./testaugmenta/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

config = Config(config_dir, learning=True)
aug=config.learning_config.augmentations

with open(file_path, "r", encoding="utf-8") as lines:
    wav = [line.split("\t", 2)[0] for line in lines]
    for i in wav:
        if i == 'PATH' :continue
        name = i.split('/')[-1]
        signal, sample_rate = load_signal(i,sample_rate)
        signal = aug.before.augment(signal)
        soundfile.write(output_path + name, signal, 16000)