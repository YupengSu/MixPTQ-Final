```
## Nvidia A800 80G GPU + CUDA 12.4 + miniconda

conda create -n mixptq python=3.10

conda activate mixptq

pip3 install torch torchvision torchaudio

pip3 install datasets transformers accelerate lm_eval texttable toml 