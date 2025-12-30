#!/usr/bin/env bash
set -e

export HF_HOME=/projets/Ymohammadi/Video/mochi/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export DIFFUSERS_CACHE=$HF_HOME/diffusers

mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"

python inference.py