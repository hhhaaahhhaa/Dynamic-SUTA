import os
import torch
from torch.utils.data import Dataset
import yaml
from tqdm import tqdm
import json

from ..system.suta import SUTASystem
from ..utils.tool import wer
from .base import IStrategy


class AWMCStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.anchor = SUTASystem(config["system_config"])
        self.leader = SUTASystem(config["system_config"])
        self.system = SUTASystem(config["system_config"])  # chaser

        # setup anchor and leader
        self.anchor.eval()
        self.leader.eval()
        for param in self.anchor.model.parameters():
            param.detach_()
        for param in self.leader.model.parameters():
            param.detach_()

        self.ema_task_vector = None
        self.alpha = 0.999
        self.opt_param_names = self.anchor.opt_param_names

        # log
        self.transcriptions = []

    @torch.no_grad
    def _update_leader(self):
        origin_model_state = self.system.history["init"][0]
        task_vector = self._get_task_vector(leader=False)

        if self.ema_task_vector is None:
            self.ema_task_vector = {}
            for name in self.opt_param_names:
                self.ema_task_vector[name] = (1 - self.alpha) * task_vector[name]
        else:
            for name in self.opt_param_names:
                self.ema_task_vector[name] = self.alpha * self.ema_task_vector[name] + (1 - self.alpha) * task_vector[name]
        
        # add back to origin model
        merged_model_state = {}
        for name in origin_model_state:
            if name in self.opt_param_names:
                merged_model_state[name] = origin_model_state[name] + self.ema_task_vector[name]
            else:
                merged_model_state[name] = origin_model_state[name]

        self.leader.history["merged"] = (merged_model_state, None, None)
        self.leader.load_snapshot("merged")

    @torch.no_grad
    def _get_task_vector(self, leader=False):
        if leader:
            model_state = self.leader.model.state_dict()
        else:
            model_state = self.system.model.state_dict()
        origin_model_state = self.system.history["init"][0]
        task_vector = {
            name: model_state[name] - origin_model_state[name]
        for name in self.opt_param_names}
        return task_vector
    
    def _update(self, sample):  # AWMC use PL instead of unsupervised objectives
        anchor_pl_target = self.anchor.inference([sample["wav"]])[0]
        self.transcriptions[-1]["anchor"] = anchor_pl_target
        self.system.train()
        is_collapse = False
        for _ in range(self.strategy_config["steps"]):
            leader_pl_target = self.leader.inference([sample["wav"]])[0]
            record = {}
            self.system.ctc_adapt_auto(
                wavs=[sample["wav"], sample["wav"]],
                texts=[anchor_pl_target, leader_pl_target],
                batch_size=1,
                record=record
            )
            if record.get("collapse", False):
                is_collapse = True
            self._update_leader()
        
        self.transcriptions[-1]["leader"] = leader_pl_target
        if is_collapse:
            print("oh no")
    
    def run(self, ds: Dataset):
        long_cnt = 0
        n_words = []
        errs, losses = [], []
        transcriptions = []
        # task_boundaries = getattr(ds, "task_boundaries", [])  # Origin boundary known setting
        task_boundaries = []
        for idx, sample in tqdm(enumerate(ds), total=len(ds)):
            if len(sample["wav"]) > 20 * 16000:
                long_cnt += 1
                continue
            self.transcriptions.append({})
            n_words.append(len(sample["text"].split(" ")))

            # update
            if idx in task_boundaries:
                print("========== reset ==========")
                self.system.load_snapshot("init")
                self.leader.load_snapshot("init")
                self.ema_task_vector = None
            self._update(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])[0]
            err = wer(sample["text"], trans)
            errs.append(err)
            transcriptions.append((sample["text"], trans))
            self.transcriptions[-1]["system"] = trans
            self.transcriptions[-1]["gt"] = sample["text"]
        
        # additional log
        os.makedirs(self.config["output_dir"]["log_dir"], exist_ok=True)
        with open(f"{self.config['output_dir']['log_dir']}/log.json", "w") as f:
            json.dump(self.transcriptions, f, indent=4)
        print("#Too long: ", long_cnt)

        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count

    def load_checkpoint(self, path):
        self.system.load(path)
