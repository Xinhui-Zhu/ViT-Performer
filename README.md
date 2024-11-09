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

## Thanks To

https://github.com/kentaroy47/vision-transformers-cifar10

