# OpenGait on CASIA-B: Setup, Training and Testing

Step-by-step guide to train and evaluate gait recognition models (Baseline / GaitSet) with [OpenGait](https://github.com/ShiqiYu/OpenGait) on the CASIA-B dataset.

## Requirements

- Linux or WSL2 (NCCL multi-GPU does not work on native Windows)
- NVIDIA GPU with a recent driver and CUDA
- Anaconda or Miniconda
- CASIA-B silhouette dataset (access must be requested)


## 1. Install Anaconda

Download from https://www.anaconda.com/download

## 2. Create the environment

```bash
conda create -n opengait python=3.8 -y
conda activate opengait
```

## 3. Clone OpenGait

```bash
git clone https://github.com/ShiqiYu/OpenGait.git
cd OpenGait
```

## 4. Install PyTorch, then the other dependencies

Install PyTorch first, using the command from https://pytorch.org that matches your CUDA version. Quote any version specifier, otherwise the shell treats `>=` as a redirect.

```bash
pip install "torch>=1.10" torchvision
pip install -r requirements.txt
```


Verify that the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

This should print `True`.

