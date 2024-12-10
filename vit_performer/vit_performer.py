import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import vit_pytorch
from vits import ViT as ViTHacked
from performers import Performer  # hacked version, as we need to modify the code from performer_pytorch
import wandb
import os
from datasets import load_dataset
import PIL
import time
import math
from torch.utils.data import Subset, DataLoader, Dataset

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
parser = argparse.ArgumentParser(description='PyTorch ViT Example with Config')
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

# Load config from YAML
config = load_config(args.config)
print(config)
use_cuda = not config["device"]["no_cuda"] and torch.cuda.is_available()
use_mps = not config["device"]["no_mps"] and torch.backends.mps.is_available()
torch.manual_seed(config["device"]["seed"])
device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
device_ids = list(range(torch.cuda.device_count())) if use_cuda else None
if config['wandb']['use_wandb']:
  wandb.init(
      project=config['wandb']['project'],
      config=config)

dataset_name = config['dataset']['name']

class ImageNetDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Ensure 3 channels
        if image.shape[0] == 1:  # Handle grayscale images
            image = image.expand(3, -1, -1)

        # Apply target transforms if needed
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

if dataset_name =='TinyImageNet':
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # Wrap ImageNet dataset
    train_dataset = ImageNetDataset(
        dataset=datasets.ImageFolder('./data/tiny-imagenet-200/train_organized', transform=train_transform)
    )
    test_dataset = ImageNetDataset(
        dataset=datasets.ImageFolder('./data/tiny-imagenet-200/val_organized', transform=test_transform)
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=12, pin_memory=True, prefetch_factor=4)

else:
    transform_list = []
    for t in config["dataset"]["transform"]:
        if t["type"] == "ToTensor":
            transform_list.append(transforms.ToTensor())
        elif t["type"] == "Normalize":
            transform_list.append(transforms.Normalize(mean=t["mean"], std=t["std"]))
    transform = transforms.Compose(transform_list)


    transform_list = []
    for t in config['dataset']['transform']:
        if t['type'] == "ToTensor":
            transform_list.append(transforms.ToTensor())
        elif t['type'] == "Normalize":
            transform_list.append(transforms.Normalize(mean=t['mean'], std=t['std']))
    transform = transforms.Compose(transform_list)

    if "imagenet" in dataset_name.lower():
        dataset = load_dataset(dataset_name, cache_dir='./data')
        def apply_transform(batch):
            batch['image'] = [transform(image.convert("RGB")) for image in batch['image']]
            return batch
        train_dataset = dataset['train'].map(apply_transform, batched=True)
        test_dataset = dataset['test'].map(apply_transform, batched=True)
    else:
        dataset_class = getattr(datasets, dataset_name)
        print(dataset_class)
        if dataset_name == "Places365":
            train_dataset = dataset_class('./data', split="train-standard", small=config['dataset']['small'], transform=transform)
            test_dataset = dataset_class('./data', split="val", small=config['dataset']['small'], transform=transform)
        else:
            train_dataset = dataset_class('./data', train=True, transform=transform)
            test_dataset = dataset_class('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["train"]["test_batch_size"], shuffle=False)

class LearnableReLU1(nn.Module):
    def __init__(self, negative_slope_init=0.01):
        super(LearnableReLU1, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(negative_slope_init, dtype=torch.float32))

    def forward(self, x):
        # Element-wise maximum with scalar 0
        return torch.maximum(torch.tensor(0.0, device=x.device), x)


class LearnableReLU2(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim or dim * 2

        # Use placeholders; layers will be dynamically adjusted
        self.linear1 = None
        self.linear2 = None

    def forward(self, x):
        # Dynamically initialize layers on the first pass
        if self.linear1 is None:
            input_dim = x.shape[-1]
            self.linear1 = nn.Linear(input_dim, input_dim).to(x.device)
        x = self.linear1(x)
        x = F.relu(x)
        return x

class LearnableSigmoid1(nn.Module):
    def __init__(self, slope_init=1.0):
        super(LearnableSigmoid1, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(slope_init, dtype=torch.float32))

    def forward(self, x):
        # Learnable sigmoid: σ(x) = 1 / (1 + exp(-α * x))
        return 1 / (1 + torch.exp(-self.alpha * x))

class LearnableSigmoid2(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super(LearnableSigmoid2, self).__init__()
        self.hidden_dim = hidden_dim or dim * 2

        # Placeholder for dynamic initialization
        self.linear1 = None
        self.linear2 = None

    def forward(self, x):
        # Dynamically initialize layers on the first forward pass
        if self.linear1 is None:
            input_dim = x.shape[-1]
            self.linear1 = nn.Linear(input_dim, self.hidden_dim).to(x.device)
            self.linear2 = nn.Linear(self.hidden_dim, input_dim).to(x.device)

        # Forward pass through the learnable layers and sigmoid
        x = self.linear1(x)
        x = F.sigmoid(x)
        x = self.linear2(x)
        return x

class LearnableExp1(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.exp(self.alpha * x + self.beta)


class LearnableExp2(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim or dim * 2

        # Use placeholders; layers will be dynamically adjusted
        self.linear1 = None

    def forward(self, x):
        # Dynamically initialize layers on the first pass
        if self.linear1 is None:
            input_dim = x.shape[-1]
            self.linear1 = nn.Linear(input_dim, input_dim).to(x.device)
        x = self.linear1(x)
        x = torch.exp(x)
        return x



kernel_fn_dict = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh(),
    "exp": torch.exp,
    "cos": torch.cos,
    "Softplus": nn.Softplus(),
    "LeakyReLU": nn.LeakyReLU(),
    "GELU": nn.GELU(),
    "ELU": nn.ELU(),
    "LearnableReLU1":LearnableReLU1(),
    "LearnableReLU2":LearnableReLU2(config['model']['dim_head']),
    "LearnableSigmoid1":LearnableSigmoid1(),
    "LearnableSigmoid2":LearnableSigmoid2(config['model']['dim_head']),
    "LearnableExp1":LearnableExp1(),
    "LearnableExp2":LearnableExp2((config['model']['dim_head']))
}

kernel_function = kernel_fn_dict.get(config['model'].get('kernel_fn', 'Softmax'))


if config['model'].get('use_performer', True):
  print("Using ViT-Performer now...")
  print(f"Using {config['model']['kernel_fn']}-Performer now...")
  if config['model']['kernel_fn']=="Softmax":
      performer = Performer(
          dim=config['model']['model_dim'],
          depth=config['model']['model_depth'],
          heads=config['model']['heads'],
          dim_head=config['model']['dim_head'],
          causal=False,
          generalized_attention=False,  # to trigger kernel use in https://github.com/lucidrains/performer-pytorch/blob/fc8b78441b1e27eb5d9b01fc738a8772cee07127/performer_pytorch/performer_pytorch.py#L263
          proj_type=config['model']['proj_type'],  # should be one of ['disabled', 'default', 'learnable']
          auto_check_redraw=(config['model']['proj_type'] == 'default'),  # do not activate auto redraw if proj matrix is learned or disabled
          ff_dropout = config['model']['dropout'],
          attn_dropout = config['model']['dropout'],
          feature_redraw_interval = config['model']['redraw_interval'],
          nb_features = config['model']['random_features'],
          ff_mult = config['model']['feedforward_dim_multiplier'],
          qkv_bias = True,
          no_projection = False, # False turns performer into transformer
      )
  else:
      performer = Performer(
          dim=config['model']['model_dim'],
          depth=config['model']['model_depth'],
          heads=config['model']['heads'],
          dim_head=config['model']['dim_head'],
          causal=False,
          generalized_attention=True,  # to trigger kernel use in https://github.com/lucidrains/performer-pytorch/blob/fc8b78441b1e27eb5d9b01fc738a8772cee07127/performer_pytorch/performer_pytorch.py#L263
          kernel_fn=kernel_function,
          proj_type=config['model']['proj_type'],  # should be one of ['disabled', 'default', 'learnable']
          auto_check_redraw=(config['model']['proj_type'] == 'default'),  # do not activate auto redraw if proj matrix is learned or disabled
          ff_dropout = config['model']['dropout'],
          attn_dropout = config['model']['dropout'],
          feature_redraw_interval = config['model']['redraw_interval'],
          nb_features = config['model']['random_features'],
          ff_mult = config['model']['feedforward_dim_multiplier'],
          qkv_bias = True,
          no_projection = False, # False turns performer into transformer
      )
  model = ViTHacked(
      dim=config['model']['model_dim'],
      image_size=config['model']['image_size'],
      patch_size=config['model']['patch_size'],
      num_classes=config['model']['num_classes'],
      transformer=performer
  )
else:
    print("Using ViT-Transformer now...")

    # Validate inputs
    dim = config['model']['model_dim']
    depth = config['model']['model_depth']
    heads = config['model']['heads']
    dim_head = config['model']['dim_head']
    ff_mult = int(config['model']['feedforward_dim_multiplier'])
    nb_features = config['model']['random_features']

    # Validation checks
    assert dim % heads == 0, "Model dim must be divisible by number of heads."
    assert isinstance(ff_mult, (int, float)) and ff_mult > 0, "Feedforward multiplier must be positive."
    assert isinstance(nb_features, int) and nb_features > 0, "Number of random features must be a positive integer."

    print(f"Initializing Performer with dim={dim}, depth={depth}, heads={heads}, dim_head={dim_head}, ff_mult={ff_mult}, nb_features={nb_features}")

    performer = Performer(
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        causal=False,
        generalized_attention=True,
        proj_type=config['model']['proj_type'],
        auto_check_redraw=(config['model']['proj_type'] == 'default'),
        kernel_fn=kernel_function,
        ff_dropout=config['model']['dropout'],
        attn_dropout=config['model']['dropout'],
        feature_redraw_interval=config['model']['redraw_interval'],
        nb_features=nb_features,
        ff_mult=ff_mult,
        qkv_bias=True,
        no_projection=True,
    )

    print(f"Initializing ViTHacked with image_size={config['model']['image_size']}, patch_size={config['model']['patch_size']}, num_classes={config['model']['num_classes']}, dim={dim}")
    model = ViTHacked(
        dim=dim,
        image_size=config['model']['image_size'],
        patch_size=config['model']['patch_size'],
        num_classes=config['model']['num_classes'],
        transformer=performer,
    )

print("Printing out the model structure:")
print(model)

# Move model to device
model = model.to(device)

# Wrap model in DataParallel if multiple GPUs are available
if use_cuda and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=device_ids)

# Optimizer and Scheduler
if config['train']['optimizer'] == "AdamW":
    optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
elif config['train']['optimizer'] == "Adadelta":
    optimizer = optim.Adadelta(model.parameters(), lr=config['train']['lr'])
else:
    raise ValueError("Unsupported optimizer specified in config")

# Scheduler
if config['train']['scheduler'] == "StepLR":
    scheduler = StepLR(optimizer, step_size=config['train']['step_size'], gamma=config['train']['gamma'])
elif config['train']['scheduler'] == "CosineAnnealingLR":
    scheduler = CosineAnnealingLR(optimizer, T_max=config['train']['T_max'])
elif config['train']['scheduler'] == None:
    scheduler = None  # No scheduler
else:
    raise ValueError(f"Unsupported scheduler specified in config: {config['train']['scheduler']}")

def train(config, model, device, train_loader, optimizer, epoch):
    model.train()
    start_time = time.time()
    total_loss = 0  # Track total loss for the epoch
    correct = 0     # Track correct predictions
    total_samples = 0  # Track total number of samples

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move data and labels to device

        # Forward pass
        optimizer.zero_grad()  # Reset gradients
        output = model(data)  # Compute model predictions


        loss = F.cross_entropy(output, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update model parameters
        # Accumulate metrics
        total_loss += loss.item() * data.size(0)  # Multiply by batch size to sum total loss
        pred = output.argmax(dim=1, keepdim=True)  # Get predictions
        correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions
        total_samples += data.size(0)  # Track total samples

        # Log progress every 50 batches
        #print(batch_idx)
        if batch_idx % 100 == 0:
            batch_accuracy = 100. * correct / total_samples
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item():.6f}, Accuracy: {batch_accuracy:.2f}%")
        # WandB logging
        if config['wandb']['use_wandb']:
            wandb.log({
                "train_loss": loss.item(),
                "train_accuracy": 100. * correct / total_samples,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })

    # Epoch summary
    epoch_loss = total_loss / total_samples  # Normalize total loss
    epoch_accuracy = 100. * correct / total_samples  # Compute epoch accuracy
    print(f"Train Epoch: {epoch} completed. Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    end_time = time.time()
    train_duration = end_time - start_time
    print(f'Epoch {epoch} | ', end="")
    return train_duration

def test(config, model, device, test_loader, epoch):
    model.eval()
    start_time = time.time()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # Forward pass
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # Sum total loss
            pred = output.argmax(dim=1, keepdim=True)  # Get predictions
            correct += pred.eq(target.view_as(pred)).sum().item()  # Count correct predictions

    # Normalize loss and calculate accuracy
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)")

    # WandB logging
    if config['wandb']['use_wandb']:
        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": accuracy,
        })

    end_time = time.time()
    test_duration = end_time - start_time
    return test_duration

# Training loop
ttl_train_duration, ttl_test_duration = 0, 0
for epoch in range(1, config['train']['epochs'] + 1):
    ttl_train_duration += train(config, model, device, train_loader, optimizer, epoch)
    ttl_test_duration += test(config, model, device, test_loader, epoch)
    if scheduler:
      scheduler.step()

log_train_speed = math.log2(ttl_train_duration / len(train_loader.dataset) / epoch)
log_inference_speed = math.log2(ttl_test_duration / len(test_loader.dataset) / epoch)
train_duration_per_epoch = ttl_train_duration / epoch
print(f"Log_2(T) training speed: {log_train_speed:.2f} sec, inference speed: {log_inference_speed:.2f} sec")
print(f"Avg training speed per epoch: {train_duration_per_epoch:.2f} S")


if config['train']['save_model']:
    if not os.path.exists("./models"):
        os.makedirs("./models")
    torch.save(model.state_dict(), "./models/{}.pt".format(config['dataset']['name']))

if config['wandb']['use_wandb']:
    wandb.finish()
