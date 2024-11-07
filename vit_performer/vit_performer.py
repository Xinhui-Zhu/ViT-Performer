import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from vit_pytorch.efficient import ViT
from performer_pytorch import Performer
import wandb
import os

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Argument for config file
    parser = argparse.ArgumentParser(description='PyTorch ViT Example with Config')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config from YAML
    config = load_config(args.config)

    # Initialize wandb if enabled in config
    if config['wandb']['use_wandb']:
        wandb.init(
            project=config['wandb']['project'],
            config=config
        )

    # Set up device
    use_cuda = not config['device']['no_cuda'] and torch.cuda.is_available()
    use_mps = not config['device']['no_mps'] and torch.backends.mps.is_available()
    torch.manual_seed(config['device']['seed'])
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")

    # Set up dataset and data loaders
    dataset_name = config['dataset']['name']
    transform_list = []
    for t in config['dataset']['transform']:
        if t['type'] == "ToTensor":
            transform_list.append(transforms.ToTensor())
        elif t['type'] == "Normalize":
            transform_list.append(transforms.Normalize(mean=t['mean'], std=t['std']))
    transform = transforms.Compose(transform_list)
    
    dataset_class = getattr(datasets, dataset_name)
    if dataset_name=="Places365":
        train_dataset = dataset_class('./data', split="train-standard", small=config['dataset']['small'], transform=transform)
        test_dataset = dataset_class('./data', split="val", small=config['dataset']['small'], transform=transform)
    else:
        train_dataset = dataset_class('./data', train=True, transform=transform)
        test_dataset = dataset_class('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['train']['test_batch_size'], shuffle=False)

    # Model configuration
    kernel_fn_dict = {
        "ReLU": nn.ReLU(),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
    }
    kernel_function = kernel_fn_dict.get(config['model'].get('kernel_fn', 'ReLU'))
    performer = Performer(
        dim=config['model']['model_dim'],
        depth=config['model']['model_depth'],
        heads=config['model']['heads'],
        causal=False,
        dim_head=config['model']['dim_head'],
        kernel_fn=kernel_function
    )
    model = ViT(
        dim=config['model']['model_dim'],
        image_size=config['model']['image_size'],
        patch_size=config['model']['patch_size'],
        num_classes=config['model']['num_classes'],
        transformer=performer
    )
    model = model.to(device)

    # Optimizer and Scheduler
    if config['train']['optimizer'] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    elif config['train']['optimizer'] == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=config['train']['lr'])
    else:
        raise ValueError("Unsupported optimizer specified in config")

    if config['train']['scheduler'] == "StepLR":
        scheduler = StepLR(optimizer, step_size=1, gamma=config['train']['gamma'])
    elif config['train']['scheduler'] == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=config['train']['T_max'])
    else:
        raise ValueError("Unsupported scheduler specified in config")

    # Training loop
    for epoch in range(1, config['train']['epochs'] + 1):
        train(config, model, device, train_loader, optimizer, epoch)
        test(config, model, device, test_loader)
        scheduler.step()

    if config['train']['save_model']:
        if not os.path.exists("./models"):
            os.makedirs("./models") 
        torch.save(model.state_dict(), "./models/{}.pt".format(config['dataset']['name']))

    if config['wandb']['use_wandb']:
        wandb.finish()

def train(config, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.expand(-1, 3, -1, -1)
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Log loss to wandb
        if config['wandb']['use_wandb']:
            wandb.log({"train_loss": loss.item(), "epoch": epoch})

def test(config, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.expand(-1, 3, -1, -1)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    if config['wandb']['use_wandb']:
        wandb.log({"test_loss": test_loss, "test_accuracy": accuracy})

if __name__ == '__main__':
    main()
