import numpy as np
import torch
import random
import jiwer


def wer(a, b):
    a = jiwer.RemovePunctuation()(a)
    b = jiwer.RemovePunctuation()(b)
    return jiwer.wer(a, b, reference_transform=jiwer.wer_standardize, hypothesis_transform=jiwer.wer_standardize)


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batchify(data, batch_size, shuffle=False):
    """
    Batch generator for list data.
    """
    n_samples = len(data)
    indices = np.arange(n_samples)
    if shuffle:  # Shuffle at the start of epoch
        np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_idx = indices[start:end]
        batch_data = [data[idx] for idx in batch_idx]
        yield batch_data
