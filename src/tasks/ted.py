from torch.utils.data import Dataset
import random

from ..corpus.corpus import TEDCorpus


class RandomSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = TEDCorpus()
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
