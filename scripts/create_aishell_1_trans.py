# Copyright 2020 TalentedSoft ( Author: Shipeng XIA )

import os
import glob
import argparse
import librosa
from tqdm.auto import tqdm
import unicodedata

from tensorflow_asr.utils.utils import preprocess_paths

parser = argparse.ArgumentParser(prog="Setup Aishell_1 Transcripts")

parser.add_argument("--dir", "-d", type=str, 
                    default=None, help="Directory of dataset")

parser.add_argument("output", type=str, 
                    default=None, help="The output .tsv transcript file path")

args = parser.parse_args()

assert args.dir and args.output


args.dir = preprocess_paths(args.dir)
args.output = preprocess_paths(args.output)

transcripts_train = []
transcripts_test = []
transcripts_dev = []

text_files = glob.glob(os.path.join(args.dir, "transcript", "*.txt"))

wav_files = []

if not os.path.exists(args.output + "/wav.scp"):  # aishell-1的数据wav和transcript数量不一致？
    with open(args.output + "/wav.scp", "w", encoding="utf-8") as output:
        for root, dirs, files in os.walk(args.dir + "/wav"):
            for file in files:
                (filename,extension) = os.path.splitext(file)
                #wav_files[filename] = os.path.join(root,file)
                if "8k" not in filename:
                    wav_files.append(filename + " " + os.path.join(root, file))
        wav_files.sort()
        for line in wav_files:
            output.write(line + "\n")

for text_file in tqdm(text_files, desc="[Loading]"):
    with open(text_file, "r", encoding="utf-8") as txt:
        with open(args.output + "/wav.scp", "r", encoding="utf-8") as wavscp:
            lines = txt.read().splitlines()
            wavs = wavscp.read().splitlines()
            for line, wav in zip(lines,wavs):
                line = line.split(" ",maxsplit=1)
                wav = wav.split(" ",maxsplit=1)
                if line[0] == wav[0]:
                    print (line[0])
                    audio_file = wav[1]
                    y, sr = librosa.load(audio_file, sr=None)
                    duration = librosa.get_duration(y, sr)
                    text = unicodedata.normalize("NFC", line[1].strip().lower())
                    if "train" in wav[1]:
                        transcripts_train.append(f"{audio_file}\t{duration:.2f}\t{text}\n")
                    elif "test" in wav[1]:
                        transcripts_test.append(f"{audio_file}\t{duration:.2f}\t{text}\n")
                    else:
                        transcripts_dev.append(f"{audio_file}\t{duration:.2f}\t{text}\n")

if not os.path.exists(args.output + "/train/"):
    os.makedirs(args.output + "/train/")
    os.makedirs(args.output + "/test/")
    os.makedirs(args.output + "/dev/")

with open(args.output + "/train/transcripts.tsv", "w", encoding="utf-8") as out:
    out.write("PATH\tDURATION\tTRANSCRIPT\n")
    for line in tqdm(transcripts_train, desc="[Writing]"):
        out.write(line)

with open(args.output + "/test/transcripts.tsv", "w", encoding="utf-8") as out:
    out.write("PATH\tDURATION\tTRANSCRIPT\n")
    for line in tqdm(transcripts_test, desc="[Writing]"):
        out.write(line)

with open(args.output + "/dev/transcripts.tsv", "w", encoding="utf-8") as out:
    out.write("PATH\tDURATION\tTRANSCRIPT\n")
    for line in tqdm(transcripts_dev, desc="[Writing]"):
        out.write(line)

#os.remove(args.output + "/wav.scp")
