import os
from pathlib import Path
import numpy as np
import torch
import json
import librosa
from datasets import load_dataset
from tqdm import tqdm
import re
from builtins import str as unicode
from scipy.io import wavfile
import pickle

from . import Define


class LibriSpeechCorpus(object):
    def __init__(self) -> None:
        self.root = "_cache/LibriSpeech"
        with open(f"{self.root}/data_info.json", "r", encoding="utf-8") as f:
            self.data_info = json.load(f)
        
        # Filter out long wavs > 20s
        self.filtered_idxs = []
        for idx, query in enumerate(self.data_info):
            if query["length"] <= 20 * 16000:
                self.filtered_idxs.append(idx)

    def __len__(self):
        return len(self.filtered_idxs)
    
    def get(self, idx):
        query = self.data_info[self.filtered_idxs[idx]]
        basename = query['basename']
        wav, _ = librosa.load(f"{self.root}/wav/{basename}.wav", sr=16000)
        text = query['text']

        return {
            "id": basename,
            "wav": wav,
            "text": text
        }


class LibriSpeechCCorpus(object):
    def __init__(self, root: str) -> None:
        self.root = root
        with open(f"{self.root}/data_info.json", "r", encoding="utf-8") as f:
            self.data_info = json.load(f)
        
        # Filter out long wavs > 20s
        # tt = 0
        self.filtered_idxs = []
        for idx, query in enumerate(self.data_info):
            if query["length"] <= 20 * 16000:
                self.filtered_idxs.append(idx)
                # tt += query["length"]
        # print(tt / 16000 / 60)  # about 5hr

    def __len__(self):
        return len(self.filtered_idxs)
    
    def get(self, idx):
        query = self.data_info[self.filtered_idxs[idx]]
        basename = query['basename']
        wav, _ = librosa.load(f"{self.root}/wav/{basename}.wav", sr=16000)
        text = query['text']

        return {
            "id": basename,
            "wav": wav,
            "text": text
        }


class CHIMECorpus(object):
    def __init__(self, ascending=False):
        # Setup
        path = Define.CHIME 
        self.path = path
        
        split = ['et05_bus_real', 'et05_bus_simu', 'et05_caf_real', 'et05_caf_simu', 'et05_ped_simu', 'et05_str_real', 'et05_str_simu']
        apath = path + "/data/audio/16kHz/enhanced"
        tpath = path + "/data/transcriptions"

        file_list = []
        for s in split: 
            split_list = list(Path(os.path.join(apath, s)).glob("*.wav"))
            file_list += split_list
        
        text = []
        for f in tqdm(file_list, desc='Read text'):
            transcription = self.read_text(tpath, str(f))
            text.append(transcription)

        self.file_list, self.text = zip(*[(f_name, txt)
                                          for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])
    
    def read_text(self, tpath, file):
        txt_list = os.path.join(tpath, "".join("/".join(file.split('/')[-2:]).split(".")[:-1])+'.trn')

        with open(txt_list, 'r') as fp:
            for line in fp:
                return ' '.join(line.split(' ')[1:]).strip('\n')
    
    def get(self, index):
        wav, _ = librosa.load(self.file_list[index], sr=16000)
        return {
            "id": self.file_list[index],
            "wav": wav,
            "text": self.text[index]
        }

    def __len__(self):
        return len(self.file_list)


def preprocess_text(text):
    text = unicode(text)
    text = text.replace("i.e.", "that is")
    text = text.replace("e.g.", "for example")
    text = text.replace("Mr.", "Mister")
    text = text.replace("Mrs.", "Mistress")
    text = text.replace("Dr.", "Doctor")
    text = text.replace("-", " ")
    text = text.upper()
    text = re.sub("[^ A-Z']", "", text)
    text = ' '.join(text.split())
    
    return text


class CommonVoiceCorpus(object):

    cache_dir = "_cache/CommonVoice"

    def __init__(self, partial=True) -> None:
        if not os.path.exists(self.cache_dir):
            self.parse()
        with open(f"{self.cache_dir}/data_info.json", "r", encoding="utf-8") as f:
            basenames = json.load(f)
        self.wav_paths = []
        self.texts = []
        if partial:
            basenames = basenames[:5000]
        for basename in basenames:
            with open(f"{self.cache_dir}/text/{basename}.txt", "r", encoding="utf-8") as f:
                text = f.read()
                self.texts.append(text.strip())
            self.wav_paths.append(f"{self.cache_dir}/wav/{basename}.wav")

    def parse(self):
        basenames = []
        src_dataset = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "en",
            split="test",
            streaming=True,
            use_auth_token=True,
            trust_remote_code=True
        )
        os.makedirs(f"{self.cache_dir}/wav", exist_ok=True)
        os.makedirs(f"{self.cache_dir}/text", exist_ok=True)
        for idx, instance in tqdm(enumerate(src_dataset)):
            wav = librosa.resample(
                instance["audio"]["array"],
                orig_sr=src_dataset.features["audio"].sampling_rate,
                target_sr=16000
            )
            wavfile.write(f"{self.cache_dir}/wav/{idx:07d}.wav", 16000, (wav * 32767).astype(np.int16))
            with open(f"{self.cache_dir}/text/{idx:07d}.txt", "w", encoding="utf-8") as f:
                f.write(instance["sentence"])
            basenames.append(f"{idx:07d}")
        with open(f"{self.cache_dir}/data_info.json", "w", encoding="utf-8") as f:
            json.dump(basenames, f, indent=4)

    def __len__(self):
        return len(self.wav_paths)
    
    def get(self, idx) -> np.ndarray:
        wav, _ = librosa.load(self.wav_paths[idx], sr=16000)
        text = preprocess_text(self.texts[idx])

        return {
            "id": self.wav_paths[idx],
            "wav": wav,
            "text": text
        }


class TEDCorpus(object):

    cache_dir = "_cache/TED"

    def __init__(self) -> None:
        if not os.path.exists(self.cache_dir):
            self.parse()
        with open(f"{self.cache_dir}/data_info.json", "r", encoding="utf-8") as f:
            info = json.load(f)
        
        self.wav_paths = []
        self.texts = []
        self.speaker_ids = []
        for (basename, speaker_id) in info:
            with open(f"{self.cache_dir}/text/{basename}.txt", "r", encoding="utf-8") as f:
                text = f.read()
                self.texts.append(text.strip())
            self.wav_paths.append(f"{self.cache_dir}/wav/{basename}.wav")
            self.speaker_ids.append(speaker_id)

    def parse(self):
        basenames = []
        speaker_ids = []
        src_dataset = load_dataset(
            "LIUM/tedlium",
            "release3",
            split="test",
            streaming=True,
            use_auth_token=True,
            trust_remote_code=True
        )
        os.makedirs(f"{self.cache_dir}/wav", exist_ok=True)
        os.makedirs(f"{self.cache_dir}/text", exist_ok=True)
        cnt = 0
        for idx, instance in tqdm(enumerate(src_dataset)):
            if instance["speaker_id"] == "inter_segment_gap":
                continue
            wav = librosa.resample(
                instance["audio"]["array"],
                orig_sr=src_dataset.features["audio"].sampling_rate,
                target_sr=16000
            )
            wavfile.write(f"{self.cache_dir}/wav/{cnt:07d}.wav", 16000, (wav * 32767).astype(np.int16))
            with open(f"{self.cache_dir}/text/{cnt:07d}.txt", "w", encoding="utf-8") as f:
                f.write(instance["text"])
            basenames.append(f"{cnt:07d}")
            speaker_ids.append(instance["speaker_id"])
            cnt += 1
        with open(f"{self.cache_dir}/data_info.json", "w", encoding="utf-8") as f:
            json.dump(list(zip(basenames, speaker_ids)), f, indent=4)

    def __len__(self):
        return len(self.wav_paths)
    
    def get(self, idx) -> np.ndarray:
        wav, _ = librosa.load(self.wav_paths[idx], sr=16000)
        text = preprocess_text(self.texts[idx])

        return {
            "id": self.wav_paths[idx],
            "wav": wav,
            "text": text
        }


if __name__ == "__main__":
    corpus = LibriSpeechCCorpus(root=f"_cache/LibriSpeech-c/GS/snr=5")
