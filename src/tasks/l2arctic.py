import numpy as np
from torch.utils.data import Dataset
import random

from ..corpus.corpus import L2ArcticCorpus


class SingleAccentSequence(Dataset):
    def __init__(self, accent: str="Korean", snr_level=None) -> None:
        self.corpus = L2ArcticCorpus()
        self.idx_seq = []
        for spk in self.corpus.accent2spks[accent]:
            self.idx_seq.extend([(spk, i) for i in range(len(self.corpus.spk2files[spk]))])
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

        self.snr_level = snr_level

    # def add_noise(self, clean_wav: np.ndarray, noise: np.ndarray) -> np.ndarray:
    #     # repeat noise content if too short
    #     noiseconcat = noise
    #     while len(noiseconcat) <= len(clean_wav):
    #         noiseconcat = np.append(noiseconcat, noise)
    #     noise = noiseconcat
    #     if len(noise) > len(clean_wav):
    #         noise = noise[0:len(clean_wav)]

    #     noisy_wav = snr_mixer(clean_wav, noise, snr=self.snr_level)
    #     return noisy_wav

    def __len__(self):
        return len(self.idx_seq)
    
    def __getitem__(self, idx):
        res = self.corpus.get(*self.idx_seq[idx])

        return res
