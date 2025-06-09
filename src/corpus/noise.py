import os
import numpy as np
import torch
import json
import librosa


class MSSNSD10(object):

    type2noisefilename = {
        "AC": "AirConditioner_6",
        "AA": "AirportAnnouncements_2",
        "BA": "Babble_4",
        "CM": "CopyMachine_2",
        "MU": "Munching_3",
        "NB": "Neighbor_6",
        "SD": "ShuttingDoor_6",
        "TP": "Typing_2",
        "VC": "VacuumCleaner_1",
        "GS": None,  # Gaussian noise
    }
        
    def get(self, noise_type: str) -> np.ndarray:
        assert noise_type in self.type2noisefilename
        noise_filename = self.type2noisefilename[noise_type]
        if noise_type == "GS":
            noise = None
            # noise = np.random.randn(*clean_wav.shape)
        else:
            noise, _ = librosa.load(f"preprocess/res/{noise_filename}.wav", sr=16000)
        return noise
