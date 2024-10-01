from torch.utils.data import Dataset
import random

from ..corpus.corpus import CHIMECorpus


class RandomSequence(Dataset):
    def __init__(self) -> None:
        self.corpus = CHIMECorpus()
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class UniqueRealSequence(Dataset):
    """ Remove multiple samples with identical transcriptions """
    def __init__(self) -> None:
        self.corpus = CHIMECorpus()
        self.idx_seq = self.unique_text_subset()
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def unique_text_subset(self) -> list[int]:
        unique_text = []
        res = []
        for idx, text in enumerate(self.corpus.text):
            if text not in unique_text and "real" in str(self.corpus.file_list[idx]):
                unique_text.append(text)
                res.append(idx)
        return res

    def __len__(self):
        return len(self.idx_seq)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
    

class UniqueSimuSequence(Dataset):
    """ Remove multiple samples with identical transcriptions """
    def __init__(self) -> None:
        self.corpus = CHIMECorpus()
        self.idx_seq = self.unique_text_subset()
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def unique_text_subset(self) -> list[int]:
        unique_text = []
        res = []
        for idx, text in enumerate(self.corpus.text):
            if text not in unique_text and "simu" in str(self.corpus.file_list[idx]):
                unique_text.append(text)
                res.append(idx)
        return res

    def __len__(self):
        return len(self.idx_seq)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])
