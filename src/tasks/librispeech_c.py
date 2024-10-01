from torch.utils.data import Dataset, Subset, ConcatDataset
import random

from ..corpus.corpus import LibriSpeechCCorpus


class RandomSequence(Dataset):
    def __init__(self, noise_type: str, snr_level=10) -> None:
        root = f"_cache/LibriSpeech-c/{noise_type}/snr={snr_level}"
        self.corpus = LibriSpeechCCorpus(root=root)
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.get(self.idx_seq[idx])


class ConcatSequence(Dataset):
    def __init__(self, datasets: list[Dataset], tid_seq: list[int], tidx_seq: list[int], task_boundaries: list[int]=[]) -> None:
        self.datasets = datasets
        self.tid_seq = tid_seq
        self.tidx_seq = tidx_seq
        self.task_boundaries = task_boundaries

    def __len__(self):
        return len(self.tidx_seq)
    
    def __getitem__(self, idx):
        tid = self.tid_seq[idx]
        tidx = self.tidx_seq[idx]

        return {
            "tid": tid,
            **self.datasets[tid].__getitem__(tidx)
        }


class MDEASY1(Dataset):
    def __new__(cls):
        datasets = [
            RandomSequence("AC", 5),
            RandomSequence("CM", 5),
            RandomSequence("TP", 5),
            RandomSequence("AA", 5),
            RandomSequence("SD", 5),
        ]
        tid_seq = []
        for i in range(len(datasets)):
            tid_seq.extend([i] * 500)
        tidx_seq = list(range(500)) * 5
        task_boundaries = [500 * i for i in range(1, 5)]

        return ConcatSequence(datasets, tid_seq, tidx_seq, task_boundaries=task_boundaries)


class MDEASY2(Dataset):
    def __new__(cls):
        datasets = [
            RandomSequence("AC", 5),
            RandomSequence("CM", 5),
            RandomSequence("TP", 5),
            RandomSequence("AA", 5),
            RandomSequence("SD", 5),
        ]
        tid_seq = []
        for i in range(len(datasets)):
            tid_seq.extend([i] * 100)
        tid_seq = tid_seq * 5

        tidx_seq = []
        st = 0
        for _ in range(5):
            tidx_seq.extend(list(range(st, st+100)) * 5)
            st += 100
        task_boundaries = [100 * i for i in range(1, 25)]
        
        return ConcatSequence(datasets, tid_seq, tidx_seq, task_boundaries=task_boundaries)
    

class MDEASY3(Dataset):
    def __new__(cls):
        datasets = [
            RandomSequence("AC", 5),
            RandomSequence("CM", 5),
            RandomSequence("TP", 5),
            RandomSequence("AA", 5),
            RandomSequence("SD", 5),
        ]
        tid_seq = []
        for i in range(len(datasets)):
            tid_seq.extend([i] * 20)
        tid_seq = tid_seq * 25

        tidx_seq = []
        st = 0
        for _ in range(25):
            tidx_seq.extend(list(range(st, st+20)) * 5)
            st += 20
        task_boundaries = [20 * i for i in range(1, 125)]

        return ConcatSequence(datasets, tid_seq, tidx_seq, task_boundaries=task_boundaries)


class MDHARD1(Dataset):
    def __new__(cls):
        datasets = [
            RandomSequence("GS", 5),
            RandomSequence("MU", 5),
            RandomSequence("VC", 5),
            RandomSequence("BA", 5),
            RandomSequence("NB", 5),
        ]
        tid_seq = []
        for i in range(len(datasets)):
            tid_seq.extend([i] * 500)
        tidx_seq = list(range(500)) * 5
        task_boundaries = [500 * i for i in range(1, 5)]

        return ConcatSequence(datasets, tid_seq, tidx_seq, task_boundaries=task_boundaries)


class MDHARD2(Dataset):
    def __new__(cls):
        datasets = [
            RandomSequence("GS", 5),
            RandomSequence("MU", 5),
            RandomSequence("VC", 5),
            RandomSequence("BA", 5),
            RandomSequence("NB", 5),
        ]
        tid_seq = []
        for i in range(len(datasets)):
            tid_seq.extend([i] * 100)
        tid_seq = tid_seq * 5

        tidx_seq = []
        st = 0
        for _ in range(5):
            tidx_seq.extend(list(range(st, st+100)) * 5)
            st += 100
        task_boundaries = [100 * i for i in range(1, 25)]

        return ConcatSequence(datasets, tid_seq, tidx_seq, task_boundaries=task_boundaries)
    

class MDHARD3(Dataset):
    def __new__(cls):
        datasets = [
            RandomSequence("GS", 5),
            RandomSequence("MU", 5),
            RandomSequence("VC", 5),
            RandomSequence("BA", 5),
            RandomSequence("NB", 5),
        ]
        tid_seq = []
        for i in range(len(datasets)):
            tid_seq.extend([i] * 20)
        tid_seq = tid_seq * 25

        tidx_seq = []
        st = 0
        for _ in range(25):
            tidx_seq.extend(list(range(st, st+20)) * 5)
            st += 20
        task_boundaries = [20 * i for i in range(1, 125)]

        return ConcatSequence(datasets, tid_seq, tidx_seq, task_boundaries=task_boundaries)


class MDLong(Dataset):
    def __new__(cls):
        datasets = [
            RandomSequence("AA", 5),
            RandomSequence("AC", 5),
            RandomSequence("BA", 5),
            RandomSequence("CM", 5),
            RandomSequence("GS", 5),
            RandomSequence("MU", 5),
            RandomSequence("NB", 5),
            RandomSequence("SD", 5),
            RandomSequence("TP", 5),
            RandomSequence("VC", 5),
        ]
        record = []
        tid_seq, tidx_seq = [], []
        task_boundaries = []
        while len(tid_seq) < 10000:
            tid = random.choice(list(range(10)))
            while tid_seq and tid == tid_seq[-1]:
                tid = random.choice(list(range(10)))
            l_seg = random.choice(list(range(20, 501)))
            tid_seq.extend([tid] * l_seg)
            idxs = random.sample(list(range(len(datasets[tid]))), k=l_seg)
            tidx_seq.extend(idxs)
            task_boundaries.append(len(tidx_seq))
            record.append((tid, len(tidx_seq)))
        tid_seq = tid_seq[:10000]
        tidx_seq = tidx_seq[:10000]
        task_boundaries[-1] = 10000
        print(record)

        return ConcatSequence(datasets, tid_seq, tidx_seq, task_boundaries=task_boundaries)


class LongIIDSequence1(Dataset):
    def __init__(self):
        datasets = [
            Subset(RandomSequence("GS", 5), list(range(500))),
            Subset(RandomSequence("MU", 5), list(range(500))),
            Subset(RandomSequence("VC", 5), list(range(500))),
            Subset(RandomSequence("BA", 5), list(range(500))),
            Subset(RandomSequence("NB", 5), list(range(500))),
        ]
        self.corpus = ConcatDataset(datasets)
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.__getitem__(self.idx_seq[idx])


class LongIIDSequence2(Dataset):
    def __init__(self):
        datasets = [
            Subset(RandomSequence("AC", 5), list(range(500))),
            Subset(RandomSequence("CM", 5), list(range(500))),
            Subset(RandomSequence("TP", 5), list(range(500))),
            Subset(RandomSequence("AA", 5), list(range(500))),
            Subset(RandomSequence("MU", 5), list(range(500))),
        ]
        self.corpus = ConcatDataset(datasets)
        self.idx_seq = list(range(len(self.corpus)))
        self.task_boundaries = []
        random.shuffle(self.idx_seq)

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx):
        return self.corpus.__getitem__(self.idx_seq[idx])
