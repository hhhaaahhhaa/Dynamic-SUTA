import os
import gc
import argparse
import yaml
import json
import pickle

from src.utils.tool import seed_everything
from src.strategies.load import get_strategy_cls
from src.tasks.load import get_task


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"  # https://stackoverflow.com/questions/73747731/runtimeerror-cuda-out-of-memory-how-can-i-set-max-split-size-mb


def create_config(args):
    """ Create a dictionary for full configuration """
    res = {
        "exp_name": args.exp_name,
        "strategy_name": args.strategy_name,
        "task_name": args.task_name,
    }
    if args.checkpoint is not None:
        res["checkpoint"] = args.checkpoint
    system_config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    res["system_config"] = system_config
    res["strategy_config"] = {}
    for path in args.strategy_config:
        strategy_config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        res["strategy_config"].update(strategy_config)

    return res


def main(args):
    config = create_config(args)

    exp_root = f"results/benchmark/{args.strategy_name}/{args.exp_name}/{args.task_name}"
    os.makedirs(exp_root, exist_ok=True)
    config["output_dir"] = {
        "log_dir": f"{exp_root}/log",
        "result_dir": f"{exp_root}/result",
        "ckpt_dir": f"{exp_root}/ckpt"
    }
    with open(f"{exp_root}/config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    strategy = get_strategy_cls(args.strategy_name)(config)
    task = get_task(args.task_name)

    if config.get("checkpoint", None) is not None:
        try:
            print(f'Load from {config["checkpoint"]}...')
            strategy.load_checkpoint(config["checkpoint"])
        except:
            print(f'Strategy {config["strategy_name"]} fails/unsupports checkpoint loading.')
            exit()

    print("========================== Start! ==========================")
    print("Exp name: ", config["exp_name"])
    print("Strategy name: ", config["strategy_name"])
    print("Task name: ", config["task_name"])
    print("Log directory: ", config["output_dir"]["log_dir"])
    print("Result directory: ", config["output_dir"]["result_dir"])
    print("Checkpoint directory: ", config["output_dir"]["ckpt_dir"])

    results = strategy.run(task)

    assert len(results["n_words"]) == len(results["wers"])  # make sure correctness
    err = 0
    for i in range(len(results["n_words"])):
        err += results["wers"][i] * results["n_words"][i]
    denom = sum(results["n_words"])
    wer = err / denom

    # log
    print(f"WER: {wer * 100:.2f}%")
    print("Step count: ", strategy.get_adapt_count())
    os.makedirs(config["output_dir"]["log_dir"], exist_ok=True)
    os.makedirs(config["output_dir"]["result_dir"], exist_ok=True)
    with open(f'{exp_root}/results.txt', "w") as f:
        f.write(f"WER: {wer * 100:.2f}%\n")
        f.write(f"Step count: {strategy.get_adapt_count()}\n")
    with open(f'{config["output_dir"]["result_dir"]}/results.pkl', "wb") as f:
        pickle.dump(results, f)

    # txt log
    if "transcriptions" in results:
        with open(f'{config["output_dir"]["log_dir"]}/transcriptions.txt', "w") as f:
            if "basenames" in results:
                for (orig, pred), wer, basename in zip(results["transcriptions"], results["wers"], results["basenames"]):
                    f.write(f"{wer * 100:.2f}%|{basename}|{orig}|{pred}\n")
            else:
                for (orig, pred), wer in zip(results["transcriptions"], results["wers"]):
                    f.write(f"{wer * 100:.2f}%|{orig}|{pred}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTA ASR")
    parser.add_argument('-s', '--strategy_name', type=str)
    parser.add_argument('-t', '--task_name', type=str)
    parser.add_argument('-n', '--exp_name', type=str, default="unnamed")
    parser.add_argument('-c', '--checkpoint', type=str, default=None)
    parser.add_argument('--config', type=str, default="config/system/suta.yaml")
    parser.add_argument('--strategy_config', nargs='+', default=["config/strategy/10step.yaml"])
    parser.add_argument('--run3', action="store_true", default=False)
    
    args = parser.parse_args()
    if args.run3:
        exp_name = args.exp_name
        for i in range(3):
            gc.collect()
            args.exp_name = f"{exp_name}-{i}"
            seed_everything(666)
            main(args)
    else:
        seed_everything(666)
        main(args)
