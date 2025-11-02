import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import wandb
from resnet.net import ResNet9


def get_cifar10_loaders(batch_size=128, num_workers=2):
    """
    Get CIFAR-10 train and test data loaders
    """
    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Download and load training data
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform_train
    )
    
    trainloader = DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Download and load test data
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform_test
    )
    
    testloader = DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return trainloader, testloader


def train_one_epoch(model, trainloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Log to wandb every 100 batches
        if (batch_idx + 1) % 100 == 0:
            batch_loss = running_loss / (batch_idx + 1)
            batch_acc = 100. * correct / total
            print(f'Epoch: {epoch} | Batch: {batch_idx + 1}/{len(trainloader)} | '
                  f'Loss: {batch_loss:.3f} | Acc: {batch_acc:.2f}%')
            wandb.log({
                'batch': epoch * len(trainloader) + batch_idx,
                'train_batch_loss': batch_loss,
                'train_batch_acc': batch_acc
            })
    
    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def test(model, testloader, criterion, device):
    """
    Test the model
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = test_loss / len(testloader)
    test_acc = 100. * correct / total
    
    print(f'\nTest Results: Loss: {test_loss:.3f} | Acc: {test_acc:.2f}%\n')
    
    # Log to wandb
    wandb.log({
        'test_loss': test_loss,
        'test_acc': test_acc
    })
    
    return test_loss, test_acc


def main():
    # Hyperparameters
    batch_size = 128
    num_epochs = 20
    learning_rate = 0.001  # α
    beta1 = 0.9            # β1
    beta2 = 0.999          # β2
    weight_decay = 5e-4
    num_workers = 2
    
    # Initialize wandb
    wandb.init(
        project="resnet9-cifar10",
        config={
            "architecture": "ResNet-9",
            "dataset": "CIFAR-10",
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
        }
    )
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load CIFAR-10 data
    print('Loading CIFAR-10 dataset...')
    trainloader, testloader = get_cifar10_loaders(batch_size, num_workers)
    print(f'Train samples: {len(trainloader.dataset)}, Test samples: {len(testloader.dataset)}')
    
    # Initialize model
    print('Initializing ResNet-9 model...')
    model = ResNet9(num_classes=10)
    model = model.to(device)
    
    # Watch model with wandb
    wandb.watch(model, log='all', log_freq=100)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate,      # α = 0.001
        betas=(beta1, beta2),  # (β1, β2) = (0.9, 0.999)
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[50, 75], 
        gamma=0.1
    )
    
    # Training loop
    print('Starting training...')
    best_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{num_epochs}')
        print(f'{"="*60}')
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device, epoch
        )
        
        # Test
        test_loss, test_acc = test(model, testloader, criterion, device)
        
        # Log epoch metrics to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            print(f'Saving best model with accuracy: {best_acc:.2f}%')
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, 'checkpoints/resnet9_best.pth')
    
    print(f'\nTraining completed! Best test accuracy: {best_acc:.2f}%')
    
    # Log best accuracy summary
    wandb.summary['best_test_acc'] = best_acc
    
    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()

