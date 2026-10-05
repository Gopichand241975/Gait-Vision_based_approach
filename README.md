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


## 5. Get the CASIA-B dataset

Request access: http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp

Use the silhouette version (pre-segmented images), not the raw video frames. Arrange it as:

```
CASIA-B/<subject 001-124>/<condition e.g. bg-01>/<view e.g. 000>/*.png
```

## 6. Convert the dataset to OpenGait's .pkl format

```bash
python datasets/pretreatment.py \
    --input_path CASIA-B \
    --output_path CASIA-B-pkl
```
Note: arguments differ between OpenGait versions (some need `--dataset CASIAB`). Check `datasets/CASIA-B/README.md` in the repo for the exact command.

## 7. Edit the config

Open `configs/baseline/baseline.yaml` (or `configs/gaitset/gaitset.yaml`) and set:

```yaml
data_cfg:
  dataset_root: /full/path/to/CASIA-B-pkl
```

The config already points to the partition file `datasets/CASIA-B/CASIA-B.json`. Make sure it matches your folder naming.



