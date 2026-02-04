
module load StdEnv/2023 gcc cuda/12.2 python/3.10

# python -m venv .venv
source .venv/bin/activate

# python -m ensurepip --upgrade
# python -m pip install -U pip setuptools wheel
# 
# python -m pip install -U Cython
# python -m pip install -U numpy
# 
# python -m pip install -e .

python -m pip install -U diffusers transformers accelerate