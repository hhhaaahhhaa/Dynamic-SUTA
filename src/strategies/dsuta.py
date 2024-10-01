from torch.utils.data import Dataset
import yaml
from tqdm import tqdm

from ..system.suta import SUTASystem
from ..utils.tool import wer
from .base import IStrategy


class Buffer(object):
    """ Easy implementation of buffer, use .data to access all data. """

    data: list

    def __init__(self, max_size: int=100) -> None:
        self.max_size = max_size
        self.data = []

    def update(self, x):
        self.data.append(x)
        if len(self.data) > self.max_size:
            self.data.pop(0)
    
    def clear(self):
        self.data.clear()


class DSUTAStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()

        # Set slow system
        self.slow_system = SUTASystem(config["system_config"])
        self.slow_system.eval()
        self.slow_system.snapshot("start")
        self.timestep = 0
        self.update_freq = config["strategy_config"]["update_freq"]
        self.memory = Buffer(max_size=config["strategy_config"]["memory"])

        self.system.snapshot("start")

    def _init_start(self, sample):
        self.system.load_snapshot("start")

    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False
        for _ in range(self.strategy_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")
    
    def _update(self, sample):
        self.memory.update(sample)
        if (self.timestep + 1) % self.update_freq == 0:
            self.slow_system.load_snapshot("start")
            self.slow_system.eval()
            record = {}
            self.slow_system.suta_adapt_auto(
                wavs=[s["wav"] for s in self.memory.data],
                batch_size=1,
                record=record,
            )
            if record.get("collapse", False):
                print("oh no")
            self.slow_system.snapshot("start")
            self.memory.clear()
        self.system.history["start"] = self.slow_system.history["start"]  # fetch start point from slow system
        
    def run(self, ds: Dataset):
        long_cnt = 0
        n_words = []
        errs, losses = [], []
        transcriptions = []
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            n_words.append(len(sample["text"].split(" ")))

            # update
            # if self.reset and self.timestep in ds.task_boundaries:
            #     print("Reset at boundary...")
            #     self.reset_strategy()
            self._init_start(sample)
            self._adapt(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))

            # # loss
            # loss = self.system.calc_suta_loss([sample["wav"]])
            # ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            # loss["ctc_loss"] = ctc_loss["ctc_loss"]
            # losses.append(loss)

            self._update(sample)
            self.timestep += 1

        print("#Too long: ", long_cnt)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count + self.slow_system.adapt_count

    def load_checkpoint(self, path):
        self.system.load(path)
