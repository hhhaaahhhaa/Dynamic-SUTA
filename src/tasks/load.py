import os
from torch.utils.data import Dataset

from ..utils.load import get_class_in_module


SRC_DIR = "src/tasks"

MULTIDOMAIN = {
    "md_easy": (f"{SRC_DIR}/librispeech_c.py", "MDEASY1"),
    "md_easy-100": (f"{SRC_DIR}/librispeech_c.py", "MDEASY2"),  # different transition rate
    "md_easy-20": (f"{SRC_DIR}/librispeech_c.py", "MDEASY3"),
    "md_hard": (f"{SRC_DIR}/librispeech_c.py", "MDHARD1"),
    "md_hard-100": (f"{SRC_DIR}/librispeech_c.py", "MDHARD2"),
    "md_hard-20": (f"{SRC_DIR}/librispeech_c.py", "MDHARD3"),
    "md_long": (f"{SRC_DIR}/librispeech_c.py", "MDLONG"),
}

BASIC = {
    "librispeech_random": (f"{SRC_DIR}/librispeech.py", "RandomSequence"),

    "chime_random": (f"{SRC_DIR}/chime.py", "RandomSequence"),
    "chime_real": (f"{SRC_DIR}/chime.py", "UniqueRealSequence"),
    "chime_simu": (f"{SRC_DIR}/chime.py", "UniqueSimuSequence"),

    "ted_random": (f"{SRC_DIR}/ted.py", "RandomSequence"),
}

TASK_MAPPING = {
    **BASIC,
    **MULTIDOMAIN,
}


def get_task(name) -> Dataset:
    if name.startswith("LS_"):  # e.g. LS_AA_5
        from . import librispeech_c
        types = name.split("_")
        noise_type = types[1]
        snr_level = 10
        if len(types) == 3:
            snr_level = int(types[2])
        ds = librispeech_c.RandomSequence(noise_type, snr_level=snr_level)
        return ds
    
    if name.startswith("L2_"):  # e.g. L2_Korean, L2_Korean_AA_5
        types = name.split("_")
        if len(types) == 2:
            from . import l2arctic
            accent = types[1]
            ds = l2arctic.SingleAccentSequence(accent)
        else:
            from . import l2arctic_c
            accent = types[1]
            noise_type = types[2]
            snr_level = int(types[3])
            ds = l2arctic_c.SingleAccentSequence(accent, noise_type, snr_level=snr_level)
        return ds
    
    module_path, class_name = TASK_MAPPING[name]
    return get_class_in_module(class_name, module_path)()
