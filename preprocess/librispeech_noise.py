import numpy as np
import os
import librosa
from datasets import load_dataset
from tqdm import tqdm
from scipy.io import wavfile
import json
import shutil


def snr_mixer(clean, noise, snr):
    # Normalizing to -25 dB FS
    rmsclean = (clean**2).mean()**0.5
    # print("clean(before): ", rmsclean)
    scalarclean = 10 ** (-25 / 20) / rmsclean
    # print(scalarclean)
    clean = clean * scalarclean
    rmsclean = (clean**2).mean()**0.5
    # print("clean(normalized): ", rmsclean)

    rmsnoise = (noise**2).mean()**0.5
    # print("noise(before): ", rmsnoise)
    scalarnoise = 10 ** (-25 / 20) / rmsnoise
    # print(scalarnoise)
    noise = noise * scalarnoise
    rmsnoise = (noise**2).mean()**0.5
    # print("noise(normalized): ", rmsnoise)
    # wavfile.write("normed_AC.wav", 16000, (noise * 32767).astype(np.int16))
    
    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10**(snr/20)) / rmsnoise
    # print(noisescalar)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    return noisyspeech


def libirspeech_preprocess():
    cache_dir = "_cache/LibriSpeech"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(f"{cache_dir}/wav", exist_ok=True)
    os.makedirs(f"{cache_dir}/text", exist_ok=True)

    src_dataset = load_dataset(
        "librispeech_asr",
        split="test.other",
        streaming=True,
        trust_remote_code=True
    )

    data_info = []
    for idx, instance in tqdm(enumerate(src_dataset)):
        wav = librosa.resample(
            instance["audio"]["array"],
            orig_sr=src_dataset.features["audio"].sampling_rate,
            target_sr=16000
        )
        basename = f"{idx:07d}"
        wavfile.write(f"{cache_dir}/wav/{basename}.wav", 16000, (wav * 32767).astype(np.int16))
        with open(f"{cache_dir}/text/{basename}.txt", "w", encoding="utf-8") as f:
            f.write(instance["text"])
        data_info.append({
            "basename": basename,
            "length": len(wav),
            "text": instance["text"],
        })
    with open(f"{cache_dir}/data_info.json", "w", encoding="utf-8") as f:
        json.dump(data_info, f, indent=4)


def sythesize(noise_type: str, snr_level=10):
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
    assert noise_type in type2noisefilename

    output_dir = f"_cache/LibriSpeech-c/{noise_type}/snr={snr_level}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/wav", exist_ok=True)
    os.makedirs(f"{output_dir}/text", exist_ok=True)
    noise_filename = type2noisefilename[noise_type]

    clean_dir = "_cache/LibriSpeech"
    with open(f"{clean_dir}/data_info.json", "r") as f:
        data_info = json.load(f)
    
    for query in tqdm(data_info):
        clean_wav, _ = librosa.load(f"{clean_dir}/wav/{query['basename']}.wav", sr=16000)
        if noise_type == "GS":
            noise = np.random.randn(*clean_wav.shape)
        else:
            noise, _ = librosa.load(f"preprocess/res/{noise_filename}.wav", sr=16000)

        # repeat noise content if too short
        noiseconcat = noise
        while len(noiseconcat) <= len(clean_wav):
            noiseconcat = np.append(noiseconcat, noise)
        noise = noiseconcat
        if len(noise) > len(clean_wav):
            noise = noise[0:len(clean_wav)]

        noisy_wav = snr_mixer(clean_wav, noise, snr=snr_level)
        # assert 1 == 2

        wav_path = f"{output_dir}/wav/{query['basename']}.wav"
        text_path = f"{output_dir}/text/{query['basename']}.txt"
        wavfile.write(wav_path, 16000, (noisy_wav * 32767).astype(np.int16))
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(query['text'])
    shutil.copyfile(f"{clean_dir}/data_info.json", f"{output_dir}/data_info.json")


if __name__ == "__main__":
    np.random.seed(666)
    libirspeech_preprocess()
    sythesize("GS", snr_level=5)
    sythesize("AC", snr_level=5)
    sythesize("AA", snr_level=5)
    sythesize("BA", snr_level=5)
    sythesize("CM", snr_level=5)
    sythesize("MU", snr_level=5)
    sythesize("TP", snr_level=5)
    sythesize("SD", snr_level=5)
    sythesize("NB", snr_level=5)
    sythesize("VC", snr_level=5)
