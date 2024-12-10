## Install
```
conda create --name vit python=3.9
conda activate vit
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r vit_performer/requirements.txt
```

## Run

```
cd vit_performer
python vit_performer.py --config config/config_mnist.yaml
python vit_performer.py --config config/config_cifar10.yaml
python vit_performer.py --config config/config_places365.yaml
python vit_performer.py --config config/config_imagenet.yaml
```

## Examples

As an example, we recorded all the training config and logs for experiments on CIFAR-10 dataset, please check on `Xiao-CIFAR-10` branch.

## Thanks To

https://github.com/kentaroy47/vision-transformers-cifar10

https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT

https://github.com/google-research/google-research/tree/master/performer/fast_attention



