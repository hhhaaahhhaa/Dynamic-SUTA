
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
import scipy.stats
import statistics

from ..system.suta import SUTASystem
from ..utils.tool import wer
from .base import IStrategy
from .dsuta import Buffer


class GaussianStatModel(object):
    def __init__(self, data: list[float]) -> None:
        self.mu = statistics.mean(data)
        self.std = statistics.stdev(data)
        # print(f"Gaussian model: mu={self.mu}, std={self.std}")
    
    def get_prob(self, x: float, reduction: int=1) -> float:  # reduce variance by multiple sampling
        return scipy.stats.norm(self.mu, self.std / (reduction ** 0.5)).cdf(x)
    
    def get_deviation(self, x: float, reduction: int=1) -> float:  # reduce variance by multiple sampling
        return (x - self.mu) / (self.std / (reduction ** 0.5))


class BoundaryResetSystem(object):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()
        
        self.reset_record = []
        self.timestep = 0
        self.update_freq = config["strategy_config"]["update_freq"]
        self.memory = Buffer(max_size=config["strategy_config"]["memory"])

        # task boundaries
        self.boundaries = None

        self.system.snapshot("start")

    def _update(self, sample):
        self.memory.update(sample)
        if (self.timestep + 1) % self.update_freq == 0:
            if self.is_reset():
                print("========== reset ==========")
                self.reset_record.append(self.timestep + 1)
                self.system.load_snapshot("init")
                self.system.snapshot("start")
            else:
                self.update(self.memory.data)
            self.memory.clear()

    def is_reset(self) -> bool:
        return (self.timestep + 1) in self.boundaries
    
    def update(self, data):
        self.system.load_snapshot("start")
        self.system.eval()
        record = {}
        self.system.suta_adapt_auto(
            wavs=[s["wav"] for s in data],
            batch_size=1,
            record=record,
        )
        if record.get("collapse", False):
            print("oh no")
        self.system.snapshot("start")


class FixFreqResetSystem(object):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()
        
        self.reset_record = []
        self.timestep = 0
        self.update_freq = config["strategy_config"]["update_freq"]
        self.memory = Buffer(max_size=config["strategy_config"]["memory"])

        # fix frequency reset
        self.reset_freq = self.t_reset = config["strategy_config"]["reset_freq"]

        self.system.snapshot("start")

    def _update(self, sample):
        self.memory.update(sample)
        if (self.timestep + 1) % self.update_freq == 0:
            if self.is_reset():
                print("========== reset ==========")
                self.reset_record.append(self.timestep + 1)
                self.system.load_snapshot("init")
                self.system.snapshot("start")
                self.t_reset += self.reset_freq
            else:
                self.update(self.memory.data)
            self.memory.clear()

    def is_reset(self) -> bool:
        return (self.timestep + 1) >= self.t_reset
    
    def update(self, data):
        self.system.load_snapshot("start")
        self.system.eval()
        record = {}
        self.system.suta_adapt_auto(
            wavs=[s["wav"] for s in data],
            batch_size=1,
            record=record,
        )
        if record.get("collapse", False):
            print("oh no")
        self.system.snapshot("start")


class DynamicResetSystem(object):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()
        
        self.reset_record = []
        self.timestep = 0
        self.update_freq = config["strategy_config"]["update_freq"]
        self.memory = Buffer(max_size=config["strategy_config"]["memory"])

        # create stat (gaussian model)
        self.stat_model = None
        self.K = config["strategy_config"]["K"]
        self.domain_stats = []
        self.t_last_reset = 0
        self.fail_cnt, self.patience = 0, config["strategy_config"]["patience"]

        self.system.snapshot("start")
    
    def _collect_improvement_stats(self, data: list, snapshot_name: str) -> list[float]:
        init_losses, domain_losses = [], []
        self.system.load_snapshot("init")
        self.system.eval()
        for sample in data:
            loss = self.system.calc_suta_loss([sample["wav"]])
            init_losses.append(loss["total_loss"])
        
        self.system.load_snapshot(snapshot_name)
        self.system.eval()
        for sample in data:
            loss = self.system.calc_suta_loss([sample["wav"]])
            domain_losses.append(loss["total_loss"])
        return [x - y for x, y in zip(domain_losses, init_losses)]
    
    def _build_stat_model(self, data: list):
        if self.timestep + 1 == self.t_last_reset + self.K // 2:
            self.system.snapshot("domain")
        elif self.t_last_reset + self.K // 2 <= self.timestep + 1 < self.t_last_reset + self.K:
            # Collect "loss improvement" stat for domain detector
            self.domain_stats.extend(self._collect_improvement_stats(data, "domain"))
        if self.timestep + 1 == self.t_last_reset + self.K:
            self.stat_model = GaussianStatModel(self.domain_stats)  # stat is collected from "domain"
            self.domain_stats = []
    
    def _update(self, sample):
        self.memory.update(sample)
        if (self.timestep + 1) % self.update_freq == 0:
            if self.is_reset():
                print("========== reset ==========")
                self.reset_record.append(self.timestep + 1)
                self.system.load_snapshot("init")
                self.system.snapshot("start")
                self.stat_model = None
                self.domain_stats = []
                self.t_last_reset = self.timestep + 1
                self.fail_cnt = 0
            else:
                self.update(self.memory.data)
            self.memory.clear()
        self._build_stat_model(data=[sample])

    def is_distribution_shift(self, data) -> bool:
        if self.stat_model is None:
            return False
        
        # Collect current loss improvement stat and then judged by domain detector (Gaussian)
        data_stat = statistics.mean(self._collect_improvement_stats(data, "domain"))
        # p = self.stat_model.get_prob(data_stat, reduction=len(data))
        deviation = self.stat_model.get_deviation(data_stat, reduction=len(data))
        # print(f"Stat: {data_stat}, deviation={deviation}")
        
        # if p > 0.9999366575:  # +4std
        # if p > 0.9544997361: # +2std
        return deviation > 2

    def is_reset(self) -> bool:
        if self.is_distribution_shift(self.memory.data):
            self.fail_cnt += 1
        else:
            self.fail_cnt = 0
        return self.fail_cnt == self.patience
    
    def update(self, data):
        self.system.load_snapshot("start")
        self.system.eval()
        record = {}
        self.system.suta_adapt_auto(
            wavs=[s["wav"] for s in data],
            batch_size=1,
            record=record,
        )
        if record.get("collapse", False):
            print("oh no")
        self.system.snapshot("start")


class DSUTAResetStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()

        # Set slow system
        if config["strategy_config"]["reset"] == "dynamic":
            self.slow_system_cls = DynamicResetSystem
        elif config["strategy_config"]["reset"] == "fix-freq":
            self.slow_system_cls = FixFreqResetSystem
        elif config["strategy_config"]["reset"] == "oracle":
            self.slow_system_cls = BoundaryResetSystem
        else:
            raise NotImplementedError
        self.slow_system = self.slow_system_cls(config)

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
        self.slow_system._update(sample)        
        self.system.history["start"] = self.slow_system.system.history["start"]  # fetch start point from slow system
    
    def run(self, ds: Dataset):
        long_cnt = 0
        n_words = []
        errs, losses = [], []
        transcriptions = []

        if isinstance(self.slow_system, BoundaryResetSystem):
            self.slow_system.boundaries = ds.task_boundaries
        
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            n_words.append(len(sample["text"].split(" ")))

            self._init_start(sample)
            self._adapt(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])
            err = wer(sample["text"], trans[0])
            errs.append(err)
            transcriptions.append((sample["text"], trans[0]))
            
            # loss
            # loss = self.system.calc_suta_loss([sample["wav"]])
            # ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            # loss["ctc_loss"] = ctc_loss["ctc_loss"]
            # losses.append(loss)

            self._update(sample)
            self.slow_system.timestep += 1  # synchronize time step
        
        print("#Too long: ", long_cnt)

        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
            "reset_step": self.slow_system.reset_record,
        }
     
    def get_adapt_count(self):
        return self.system.adapt_count + self.slow_system.system.adapt_count

    def load_checkpoint(self, path):
        self.system.load(path)
