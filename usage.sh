#!/bin/bash
python run_benchmark.py -s none -t LS_GS_5  # origin
python run_benchmark.py -s suta -t LS_GS_5  # suta
python run_benchmark.py -s dsuta -t LS_GS_5 --strategy_config config/strategy/dsuta.yaml config/strategy/5step.yaml  # dsuta
python run_benchmark.py -s none -t chime_random
python run_benchmark.py -s suta -t chime_random
python run_benchmark.py -s dsuta -t chime_random --strategy_config config/strategy/dsuta.yaml config/strategy/5step.yaml

python run_benchmark.py -s dsuta-reset -t md_easy -n freq=5/step=5 --strategy_config config/strategy/dynamic-reset.yaml config/strategy/5step.yaml  # dynamic reset
python run_benchmark.py -s dsuta-reset -t md_easy -n freq=5/step=5-fix-freq --strategy_config config/strategy/fix-freq-reset.yaml config/strategy/5step.yaml  # fix-freq reset
python run_benchmark.py -s dsuta-reset -t md_easy -n freq=5/step=5-oracle --strategy_config config/strategy/oracle-reset.yaml config/strategy/5step.yaml  # oracle reset

# Other
python run_benchmark.py -s csuta -t chime_random --strategy_config config/strategy/1step.yaml  # csuta
python run_benchmark.py -s awmc -t chime_random --config config/system/awmc.yaml  # AWMC
