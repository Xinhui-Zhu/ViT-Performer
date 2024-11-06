# 4540-Project-ViT

## Install
```
conda create --name vit python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r vit_performer/requirements.txt
```

## Run

```
python vit_performer/mnist_vit_performer.py --kernel_fn ReLU --dataset MNIST
```
