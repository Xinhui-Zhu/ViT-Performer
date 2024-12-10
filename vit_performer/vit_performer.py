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

            self.linear1 = None

        def forward(self, x):
            if self.linear1 is None:
                input_dim = x.shape[-1]
                self.linear1 = nn.Linear(input_dim, input_dim).to(x.device)
            x = self.linear1(x)
            x = torch.exp(x)
            return x

class LearnableReLU1(nn.Module):
        def __init__(self, negative_slope_init=0.01):
            super(LearnableReLU1, self).__init__()
            #
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.beta = nn.Parameter(torch.tensor(0.0))


        def forward(self, x):
            #print(self.alpha)
            return torch.maximum(self.alpha*x+self.beta, torch.tensor(0.0, device=x.device))

class LearnableReLU2(nn.Module):
        def __init__(self, dim, hidden_dim=None):
            super().__init__()
            self.hidden_dim = hidden_dim or dim * 2

            self.linear1 = None
            self.linear2 = None

        def forward(self, x):
            if self.linear1 is None:
                input_dim = x.shape[-1]
                self.linear1 = nn.Linear(input_dim, input_dim).to(x.device)
            x = self.linear1(x)
            x = F.relu(x)
            return x

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

    # Set up device for multi-GPU support
    use_cuda = not config['device']['no_cuda'] and torch.cuda.is_available()
    use_mps = not config['device']['no_mps'] and torch.backends.mps.is_available()
    torch.manual_seed(config['device']['seed'])
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")
    device_ids = list(range(torch.cuda.device_count())) if use_cuda else None

    # Set up dataset and data loaders
    dataset_name = config['dataset']['name']
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
        if dataset_name == "Places365":
            train_dataset = dataset_class('./data', download=False, split="train-standard", small=config['dataset']['small'], transform=transform)
            test_dataset = dataset_class('./data', download=False, split="val", small=config['dataset']['small'], transform=transform)
        else:
            train_dataset = dataset_class('./data', download=True, train=True, transform=transform)
            test_dataset = dataset_class('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['train']['test_batch_size'], shuffle=False)

    # Model configuration
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
        "Softmax": nn.Softmax(),
        "LearnableReLU1":LearnableReLU1(),
        "LearnableReLU2":LearnableReLU2((config['model']['dim_head'])),
        "LearnableExp1":LearnableExp1(),
        "LearnableExp2":LearnableExp2((config['model']['dim_head']))
    }
    
    kernel_function = kernel_fn_dict.get(config['model'].get('kernel_fn', 'Softmax'))

    if config['model'].get('use_performer', True):
        print(f"Using {config['model']['kernel_fn']}-Performer now...")
        print(f"Random feature number: {config['model']['random_features']}")
        if config['model'].get('kernel_fn') == "Softmax":
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
        performer = Performer(
            dim=config['model']['model_dim'],
            depth=config['model']['model_depth'],
            heads=config['model']['heads'],
            dim_head=config['model']['dim_head'],
            causal=False,
            generalized_attention=True,  # to trigger kernel use in https://github.com/lucidrains/performer-pytorch/blob/fc8b78441b1e27eb5d9b01fc738a8772cee07127/performer_pytorch/performer_pytorch.py#L263
            proj_type=config['model']['proj_type'],  # should be one of ['disabled', 'default', 'learnable']
            auto_check_redraw=(config['model']['proj_type'] == 'default'),  # do not activate auto redraw if proj matrix is learned or disabled
            kernel_fn=kernel_function,
            ff_dropout = config['model']['dropout'],
            attn_dropout = config['model']['dropout'],
            feature_redraw_interval = config['model']['redraw_interval'],
            nb_features = config['model']['random_features'],
            ff_mult = config['model']['feedforward_dim_multiplier'],
            qkv_bias = True,
            no_projection = True,
        )
        model = ViTHacked(
            dim=config['model']['model_dim'],
            image_size=config['model']['image_size'],
            patch_size=config['model']['patch_size'],
            num_classes=config['model']['num_classes'],
            transformer=performer
        )

    if config['model'].get("print_model", True):
        print("Printing out the model structure:")
        print("Note: The model structure may appear repeated inside 'proj_updater'.")
        print("This is intentional and does not affect the model's functionality or computational complexity.")
        print("The repeated structure is for managing projection updates and is only a reference to previous structure.")
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

    if config['train']['scheduler'] == "StepLR":
        scheduler = StepLR(optimizer, step_size=1, gamma=config['train']['gamma'])
    elif config['train']['scheduler'] == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=config['train']['T_max'])
    else:
        raise ValueError("Unsupported scheduler specified in config")

    # Training loop
    ttl_train_duration, ttl_test_duration = 0, 0
    for epoch in range(1, config['train']['epochs'] + 1):
        ttl_train_duration += train(config, model, device, train_loader, optimizer, epoch)
        ttl_test_duration += test(config, model, device, test_loader, epoch, config['model'].get('use_performer', True))
        scheduler.step()

    train_duration_per_epoch = ttl_train_duration / epoch
    test_duration_per_epoch = ttl_test_duration / epoch
    log_train_speed = math.log2(ttl_train_duration / len(train_loader.dataset) / epoch)
    log_inference_speed = math.log2(ttl_test_duration / len(test_loader.dataset) / epoch)
    print("*************************************")
    print(f"Train loader dataset length: {len(train_loader.dataset)}")
    print(f"Test loader dataset length: {len(test_loader.dataset)}")
    print(f"Log_2(T) training speed: {log_train_speed:.2f} S")
    print(f"Log_2(T) inference speed: {log_inference_speed:.2f} S")
    print(f"Avg training speed per epoch: {train_duration_per_epoch:.2f} S")
    print(f"Avg testing speed per cycle: {test_duration_per_epoch:.2f} S")
    print("*************************************")

    if config['train']['save_model']:
        if not os.path.exists("./models"):
            os.makedirs("./models") 
        torch.save(model.state_dict(), "./models/{}.pt".format(config['dataset']['name']))

    if config['wandb']['use_wandb']:
        wandb.finish()

def train(config, model, device, train_loader, optimizer, epoch):
    model.train()
    start_time = time.time()
    for batch_idx, batch in enumerate(train_loader):
        if "imagenet" in config['dataset']['name'].lower():
            target = batch['label'].clone().detach().long()
            data = batch['image']
            data = torch.stack([torch.stack([torch.stack([x.clone().detach() for x in row]) for row in channel]) for channel in data])
            data = data.permute(3, 0, 1, 2).float()    
        else:
            data, target = batch
        data, target = data.to(device), target.to(device)
        data = data.expand(-1, 3, -1, -1)  # Ensure data is 3-channel for ViT model
        optimizer.zero_grad()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Log loss to wandb
        if config['wandb']['use_wandb']:
            wandb.log({"train_loss": loss.item(), 
                       "learning_rate": optimizer.param_groups[0]['lr'],
                       "epoch": epoch - 1 + batch_idx/len(train_loader.dataset),
                       })
            
    end_time = time.time()  
    train_duration = end_time - start_time  
    print(f'Epoch {epoch} | ', end="")
    return train_duration

def test(config, model, device, test_loader, epoch, performer):
    model.eval()
    test_loss = 0
    correct = 0
    start_time = time.time()
    
    # attention_matrix_saved = False  # Flag to check if attention matrix is already saved

    # # Function to save the attention matrix
    # def save_attention_matrix(module, input, output):
    #     nonlocal attention_matrix_saved  # Access the flag from outer scope
    #     if not attention_matrix_saved:
    #         attention_matrix = output[1].detach().cpu()
    #         # Save the attention matrix to a file
    #         os.makedirs("output", exist_ok=True)
    #         torch.save(attention_matrix, os.path.join("output", f"{epoch}_attention_matrix.pt"))
    #         attention_matrix_saved = True  # Set flag to True after saving once
    
    # if performer:
    #     # Register hooks on each SelfAttention layer in the model
    #     for layer in model.transformer.net.layers:  # Access layers within transformer.net
    #         if isinstance(layer, torch.nn.ModuleList):
    #             for sublayer in layer:
    #                 # Check if the layer contains SelfAttention
    #                 if isinstance(sublayer.fn, torch.nn.Module) and hasattr(sublayer.fn, 'fast_attention'):
    #                     sublayer.fn.register_forward_hook(save_attention_matrix)
    # else:
    #     # Register hook on each Attention layer within the transformer
    #     for layer in model.transformer.layers:
    #         # Each layer is a ModuleList with [Attention, FeedForward]
    #         layer[0].attend.register_forward_hook(save_attention_matrix)

    with torch.no_grad():
        for batch in test_loader:
            # Prepare data and target
            if "imagenet" in config['dataset']['name'].lower():
                target = batch['label'].clone().detach().long()
                data = batch['image']
                data = torch.stack([torch.stack([torch.stack([x.clone().detach() for x in row]) for row in channel]) for channel in data])
                data = data.permute(3, 0, 1, 2).float()
            else:
                data, target = batch
            data, target = data.to(device), target.to(device)
            data = data.expand(-1, 3, -1, -1)
            
            # Forward pass through the model
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate and print test statistics
    end_time = time.time()
    test_duration = end_time - start_time
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')

    # Log to wandb if enabled
    if config['wandb']['use_wandb']:
        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": accuracy,
        })

    return test_duration

if __name__ == '__main__':
    main()
