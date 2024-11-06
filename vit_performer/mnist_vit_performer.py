from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from vit_pytorch.efficient import ViT
from performer_pytorch import Performer
import wandb


def train(args, model, device, train_loader, optimizer, epoch):
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
        wandb.log({"train_loss": loss.item(), "epoch": epoch})
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.expand(-1, 3, -1, -1)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    # Log test loss and accuracy to wandb
    wandb.log({"test_loss": test_loss, "test_accuracy": accuracy})


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--kernel_fn', type=str, default="ReLU",
                        help='Fast attention kernels')
    parser.add_argument('--dataset', type=str, default="MNIST",
                        help='Training and testing dataset name')
    
    args = parser.parse_args()

    wandb.init(
            project="vit-performer",
            config={
                "kernel_fn": args.kernel_fn,
                "dataset": args.dataset,
            }
        )   
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 6,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    dataset_dict = {
        "MNIST": {
            "dataset": datasets.MNIST,
            "transform": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        },
        "CIFAR10": {
            "dataset": datasets.CIFAR10,
            "transform": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        },
        "ImageNet": {
            "dataset": datasets.ImageNet,
            "transform": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        },
        "Places365": {
            "dataset": datasets.Places365,
            "transform": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
    }

    # Select dataset configuration
    dataset_info = dataset_dict.get(args.dataset)
    if dataset_info is None:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Load datasets
    dataset1 = dataset_info['dataset']('./data', train=True, download=True, transform=dataset_info["transform"])
    dataset2 = dataset_info['dataset']('./data', train=False, transform=dataset_info["transform"])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    kernel_fn_dict = {
        "ReLU": nn.ReLU(),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
    }

    # Get kernel function
    kernel_function = kernel_fn_dict.get(args.kernel_fn)
    if kernel_function is None:
        raise ValueError(f"Unsupported kernel function: {args.kernel_fn}")


    performer = Performer(
        dim = 512,
        depth = 8,
        heads = 8,
        causal = False,
        dim_head = 64,
        kernel_fn = kernel_function
    )

    model = ViT(
        dim = 512,
        image_size = 28,
        patch_size = 7,
        num_classes = 10,
        transformer = performer
    )
    
    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "./models/mnist_vit_performer.pt")

    wandb.finish()

if __name__ == '__main__':
    main()