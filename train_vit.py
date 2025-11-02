import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import os
import wandb
from vit.model import create_dinov3_model, get_transform
from tqdm import tqdm


def get_cifar10_loaders(batch_size=128, num_workers=2, img_size=448):
    """
    Get CIFAR-10 train and test data loaders with DINOv3 preprocessing
    """
    # Get transforms from vit.model
    transform_train = get_transform(img_size=img_size, is_training=True)
    transform_test = get_transform(img_size=img_size, is_training=False)
    
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


def train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, gradient_accumulation_steps=1):
    """
    Train for one epoch with gradient accumulation support
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Scale loss by accumulation steps
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Statistics (use unscaled loss for logging)
        running_loss += loss.item() * gradient_accumulation_steps
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
    
    # Final update if there are remaining gradients
    if len(trainloader) % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
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
    batch_size = 64  # Smaller batch size due to larger images (448x448)
    gradient_accumulation_steps = 1  # No gradient accumulation for classifier-only training
    num_epochs = 2
    learning_rate = 0.001  # α
    beta1 = 0.9            # β1
    beta2 = 0.999          # β2
    weight_decay = 5e-4
    num_workers = 2
    img_size = 448
    freeze_backbone = True  # FREEZE BACKBONE - only train classifier
    use_gradient_checkpointing = False  # Not needed for classifier-only training
    
    # Initialize wandb
    wandb.init(
        project="dinov3-cifar10",
        config={
            "architecture": "DINOv3-ViT-Large",
            "dataset": "CIFAR-10",
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "img_size": img_size,
            "freeze_backbone": freeze_backbone,
            "training_mode": "classifier_only"
        }
    )
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load CIFAR-10 data
    print('Loading CIFAR-10 dataset...')
    trainloader, testloader = get_cifar10_loaders(batch_size, num_workers, img_size)
    print(f'Train samples: {len(trainloader.dataset)}, Test samples: {len(testloader.dataset)}')
    
    # Initialize model
    print('Initializing DINOv3 model...')
    model = create_dinov3_model(
        num_classes=10,
        model_name='vit_large_patch16_dinov3',
        pretrained=True,
        freeze_backbone=freeze_backbone,  # Full fine-tuning or classifier only
        img_size=img_size
    )
    
    # Enable gradient checkpointing for memory efficiency during full fine-tuning
    if use_gradient_checkpointing and not freeze_backbone:
        print("Enabling gradient checkpointing to save memory...")
        if hasattr(model.backbone, 'set_grad_checkpointing'):
            model.backbone.set_grad_checkpointing(enable=True)
        elif hasattr(model.backbone, 'gradient_checkpointing_enable'):
            model.backbone.gradient_checkpointing_enable()
        else:
            print("Warning: Gradient checkpointing not available for this model")
    
    model = model.to(device)
    
    # Verify training status
    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    classifier_trainable = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    
    print(f"Backbone parameters: {backbone_params:,}")
    print(f"Backbone trainable: {backbone_trainable:,}")
    print(f"Backbone frozen: {backbone_params - backbone_trainable:,}")
    print(f"\nClassifier parameters: {classifier_params:,}")
    print(f"Classifier trainable: {classifier_trainable:,}")
    print(f"\nTotal trainable: {backbone_trainable + classifier_trainable:,}")
    print(f"\n✓ Training mode: {'FULL FINE-TUNING' if backbone_trainable > 0 else 'CLASSIFIER ONLY'}")
    print("="*60 + "\n")
    
    # Watch model with wandb
    wandb.watch(model, log='all', log_freq=100)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Optimize based on training mode
    if freeze_backbone:
        # Only optimize classifier parameters (backbone is frozen)
        print("Optimizer: Training classifier only")
        optimizer = optim.Adam(
            model.classifier.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
    else:
        # Full fine-tuning - optimize all parameters with differential learning rates
        print("Optimizer: Full fine-tuning with differential learning rates")
        optimizer = optim.Adam([
            {'params': model.backbone.parameters(), 'lr': learning_rate},  # Small LR for backbone
            {'params': model.classifier.parameters(), 'lr': learning_rate * 10}  # Larger LR for classifier
        ],
        betas=(beta1, beta2),
        weight_decay=weight_decay
        )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[10, 15], 
        gamma=0.1
    )
    
    # Training loop
    training_mode = 'FULL FINE-TUNING' if not freeze_backbone else 'CLASSIFIER ONLY'
    print(f'Starting training in {training_mode} mode...')
    if not freeze_backbone:
        print(f'Backbone LR: {learning_rate:.2e}, Classifier LR: {learning_rate * 10:.2e}')
        print(f'Gradient accumulation steps: {gradient_accumulation_steps}')
        print(f'Effective batch size: {batch_size * gradient_accumulation_steps}')
    best_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{num_epochs}')
        print(f'{"="*60}')
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device, epoch,
            gradient_accumulation_steps=gradient_accumulation_steps if not freeze_backbone else 1
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
            
            # Use different checkpoint name based on training mode
            checkpoint_name = 'dinov3_full_finetune_best.pth' if not freeze_backbone else 'dinov3_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'freeze_backbone': freeze_backbone,
            }, f'checkpoints/{checkpoint_name}')
    
    print(f'\nTraining completed! Best test accuracy: {best_acc:.2f}%')
    
    # Log best accuracy summary
    wandb.summary['best_test_acc'] = best_acc
    
    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()

