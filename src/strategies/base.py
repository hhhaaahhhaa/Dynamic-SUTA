from torch.utils.data import Dataset


class IStrategy(object):
    def run(self, ds: Dataset):
        raise NotImplementedError
    
    def get_adapt_count(self) -> int:
        return 0
