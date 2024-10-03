# Dynamic-SUTA
Official implementation of EMNLP 2024 paper [Continual Test-time Adaptation for End-to-end Speech Recognition on Noisy Speech](https://arxiv.org/abs/2406.11064).

## Installation
```
git clone https://github.com/hhhaaahhhaa/Dynamic-SUTA.git
cd Dynamic-SUTA
pip install -r requirements.txt
```

## Data Preparation
To preprocess librispeech test set and add 10 noises, see preprocess/librispeech_noise.py.
```
python -m preprocess.librispeech_noise
```
For CHIME3, set your local path in `src/corpus/Define.py`.

## Usage
```
python run_benchmark.py -s [strategy_name] -t [task_name] -n [exp_name] --config [system_config] --strategy_config [strategy_config1 ...]
```
The results are in `results/benchmark/[strategy_name]/[exp_name]/[task_name]`

Available `strategy_name` tags are in `src/strategies/load.py`, available `task_name` tags are in `src/tasks/load.py`. See `usage.sh` for more example commands.

The repo is designed to allow researchers to easily add new strategies and tasks with new tags.

## Contact
- Guan-Ting Lin [email] daniel094144@gmail.com
- Wei-Ping Huang [email] thomas1232121@gmail.com

## Citation
If you find our work useful, please use the following citation:
```
@misc{lin2024continualtesttimeadaptationendtoend,
      title={Continual Test-time Adaptation for End-to-end Speech Recognition on Noisy Speech}, 
      author={Guan-Ting Lin and Wei-Ping Huang and Hung-yi Lee},
      year={2024},
      eprint={2406.11064},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2406.11064}, 
}
```
