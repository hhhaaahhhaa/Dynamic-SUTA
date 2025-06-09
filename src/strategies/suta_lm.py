import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.special
import yaml
from tqdm import tqdm
from collections import defaultdict

from ..system.suta import SUTASystem
from ..utils.tool import wer
from .base import IStrategy


class SUTALMStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()

        self._log = None

    def _reset_state(self):
        self._state = {
            "patience_cnt": 0,
            "best_score": -2e9,
            "selected_step_idx": -1,
            "selected_logits": None,
        }
    
    def _run_auto_step_selection(self, sample, step_idx: int, is_last=False) -> bool:
        trans, logits = self.system.inference([sample["wav"]], return_logits=True)
        trans, logits = trans[0], logits[0]

        # acoustic score thresholding
        acoustic_score = np.sum(np.max(scipy.special.log_softmax(logits, axis=-1), axis=-1))
        avg_acoustic_score = acoustic_score / logits.shape[0]
        acoustic_threshold = self.strategy_config.get("acoustic_threshold", -1000)
        if is_last and self._state["selected_step_idx"] == -1:
            pass
        elif avg_acoustic_score >= acoustic_threshold:
            pass
        else:
            return False  # keep adapting
        
        # linguistic score selection
        linguistic_score = self.system.calc_lm_score(trans)
        if linguistic_score > self._state["best_score"]:
            self._state["best_score"], self._state["selected_step_idx"] = linguistic_score, step_idx
            self._state["selected_logits"] = logits
            self._state["patience_cnt"] = 0
        else:
            self._state["patience_cnt"] += 1
        if self._state["patience_cnt"] == self.strategy_config.get("patience", 2e9):
            return True  # early stop
        return False  # keep adapting

    def _init_start(self, sample) -> None:
        self.system.load_snapshot("init")

    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False

        self._reset_state()
        self._run_auto_step_selection(sample, 0) 
        for idx in range(self.strategy_config["steps"]):
            record = {}
            self.system.suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True

            is_stop = self._run_auto_step_selection(sample, idx+1, is_last=(idx == self.strategy_config["steps"]-1))
            if is_stop:
                break
        if is_collapse:
            print("oh no")

        # beam inference
        res = self.system.processor.decode(self._state["selected_logits"], n_best=5, alpha=0.5, beta=0.0)
        nbest_trans = list(res.text)

        self._log["best_steps"].append(self._state["selected_step_idx"])
        self._log["nbest_trans"].append(nbest_trans)

        return nbest_trans[0]

    def _update(self, sample):
        pass

    def run(self, ds: Dataset):
        long_cnt = 0
        self._log = defaultdict(list)
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            self._log["n_words"].append(len(sample["text"].split(" ")))

            self._init_start(sample)
            trans = self._adapt(sample)

            err = wer(sample["text"], trans)
            self._log["wers"].append(err)
            self._log["transcriptions"].append((sample["text"], trans))
            self._log["basenames"].append(sample["id"])

            self._update(sample)
            
        print("#Too long: ", long_cnt)
        
        return self._log
    
    def get_adapt_count(self):
        return self.system.adapt_count
