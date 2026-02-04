#!/usr/bin/env bash
set -e

module load StdEnv/2023 gcc cuda/12.2 python/3.10

source .venv/bin/activate

# export HF_HOME=/projets/Ymohammadi/Video/mochi/hf_cache
# export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
# export TRANSFORMERS_CACHE=$HF_HOME/transformers
# export DIFFUSERS_CACHE=$HF_HOME/diffusers

export HF_HOME=/home/ymbahram/scratch/mochi/
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export DIFFUSERS_CACHE=$HF_HOME/diffusers

mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"

python my_files/scripts/inference3.py