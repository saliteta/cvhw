import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import ConfusionMatrix, Accuracy, Precision, Recall, F1Score
import numpy as np
from resnet.net import ResNet9
from vit.model import create_dinov3_model, get_transform
import os


# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']


def get_resnet_test_loader(batch_size=128, num_workers=2):
    """
    Get CIFAR-10 test data loader for ResNet9 (32x32 images)
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
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
    
    return testloader


def get_vit_test_loader(batch_size=64, num_workers=2, img_size=448):
    """
    Get CIFAR-10 test data loader for ViT (448x448 images)
    """
    transform_test = get_transform(img_size=img_size, is_training=False)
    
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
    
    return testloader


def load_resnet9_model(checkpoint_path, device):
    """
    Load ResNet9 model from checkpoint
    """
    print(f"\nLoading ResNet9 from {checkpoint_path}...")
    model = ResNet9(num_classes=10)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    print(f"  Checkpoint test accuracy: {checkpoint['test_acc']:.2f}%")
    
    return model


def load_vit_model(checkpoint_path, device, img_size=448):
    """
    Load ViT (DINOv3) model from checkpoint
    """
    print(f"\nLoading ViT (DINOv3) from {checkpoint_path}...")
    model = create_dinov3_model(
        num_classes=10,
        model_name='vit_large_patch16_dinov3',
        pretrained=True,
        freeze_backbone=True,
        img_size=img_size
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    print(f"  Checkpoint test accuracy: {checkpoint['test_acc']:.2f}%")
    
    return model


def evaluate_model_with_metrics(model, testloader, device, num_classes=10):
    """
    Evaluate model and compute metrics using torchmetrics
    
    Returns:
        dict: Dictionary containing all predictions, targets, and computed metrics
    """
    # Initialize torchmetrics on the correct device
    confmat = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(device)
    accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='micro').to(device)
    precision = Precision(task='multiclass', num_classes=num_classes, average='macro').to(device)
    recall = Recall(task='multiclass', num_classes=num_classes, average='macro').to(device)
    f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device)
    
    # Per-class metrics
    precision_per_class = Precision(task='multiclass', num_classes=num_classes, average=None).to(device)
    recall_per_class = Recall(task='multiclass', num_classes=num_classes, average=None).to(device)
    f1_per_class = F1Score(task='multiclass', num_classes=num_classes, average=None).to(device)
    
    all_preds = []
    all_targets = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Update metrics
            confmat.update(predicted, targets)
            accuracy.update(predicted, targets)
            precision.update(predicted, targets)
            recall.update(predicted, targets)
            f1.update(predicted, targets)
            precision_per_class.update(predicted, targets)
            recall_per_class.update(predicted, targets)
            f1_per_class.update(predicted, targets)
            
            # Store for later analysis
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx + 1}/{len(testloader)} batches")
    
    # Compute final metrics
    results = {
        'confusion_matrix': confmat.compute().cpu().numpy(),
        'accuracy': accuracy.compute().item() * 100,
        'precision': precision.compute().item(),
        'recall': recall.compute().item(),
        'f1_score': f1.compute().item(),
        'precision_per_class': precision_per_class.compute().cpu().numpy(),
        'recall_per_class': recall_per_class.compute().cpu().numpy(),
        'f1_per_class': f1_per_class.compute().cpu().numpy(),
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets)
    }
    
    return results


def plot_confusion_matrix(cm, class_names, title, save_path):
    """
    Plot and save confusion matrix using seaborn
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion'},
        square=True
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved confusion matrix to {save_path}")
    plt.close()


def plot_per_class_metrics(metrics_dict, class_names, model_name, save_path):
    """
    Plot per-class precision, recall, and F1 scores
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.arange(len(class_names))
    width = 0.6
    
    # Precision
    axes[0].bar(x, metrics_dict['precision_per_class'], width, color='skyblue')
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title('Precision per Class', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[1].bar(x, metrics_dict['recall_per_class'], width, color='lightcoral')
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].set_title('Recall per Class', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis='y', alpha=0.3)
    
    # F1 Score
    axes[2].bar(x, metrics_dict['f1_per_class'], width, color='lightgreen')
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('F1 Score per Class', fontsize=14)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'{model_name} - Per-Class Metrics', fontsize=16, y=1.02)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved per-class metrics to {save_path}")
    plt.close()


def plot_comparison(metrics_dict, class_names, save_path):
    """
    Plot comparison of metrics between multiple models
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
        class_names: List of class names
        save_path: Path to save the plot
    """
    num_models = len(metrics_dict)
    model_names = list(metrics_dict.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    x = np.arange(len(class_names))
    width = 0.25  # Width for 3 models
    offsets = np.linspace(-width, width, num_models)
    
    colors = ['skyblue', 'orange', 'lightgreen']
    
    # Precision comparison
    for idx, (model_name, offset, color) in enumerate(zip(model_names, offsets, colors)):
        axes[0].bar(x + offset, metrics_dict[model_name]['precision_per_class'], 
                   width, label=model_name, color=color, alpha=0.8)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_title('Precision Comparison', fontsize=14)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Recall comparison
    colors_recall = ['lightcoral', 'red', 'darkred']
    for idx, (model_name, offset, color) in enumerate(zip(model_names, offsets, colors_recall)):
        axes[1].bar(x + offset, metrics_dict[model_name]['recall_per_class'],
                   width, label=model_name, color=color, alpha=0.8)
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].set_title('Recall Comparison', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    # F1 Score comparison
    colors_f1 = ['lightgreen', 'darkgreen', 'forestgreen']
    for idx, (model_name, offset, color) in enumerate(zip(model_names, offsets, colors_f1)):
        axes[2].bar(x + offset, metrics_dict[model_name]['f1_per_class'],
                   width, label=model_name, color=color, alpha=0.8)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('F1 Score Comparison', fontsize=14)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].set_ylim(0, 1.05)
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Model Comparison - Per-Class Metrics', fontsize=16, y=1.02)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved comparison plot to {save_path}")
    plt.close()


def print_metrics_summary(metrics, model_name):
    """
    Print a summary of all metrics
    """
    print(f"\n{'='*60}")
    print(f"{model_name} - Metrics Summary")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Macro Precision:  {metrics['precision']:.4f}")
    print(f"Macro Recall:     {metrics['recall']:.4f}")
    print(f"Macro F1 Score:   {metrics['f1_score']:.4f}")
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print(f"{'-'*48}")
    
    for i, class_name in enumerate(CIFAR10_CLASSES):
        print(f"{class_name:<12} {metrics['precision_per_class'][i]:>10.4f}  "
              f"{metrics['recall_per_class'][i]:>10.4f}  "
              f"{metrics['f1_per_class'][i]:>10.4f}")
    print(f"{'='*60}\n")


def main():
    # Configuration
    checkpoints = {
        'ResNet9': 'checkpoints/resnet9_best.pth',
        'ViT-Classifier': 'checkpoints/dinov3_best.pth',
        'ViT-FullFT': 'checkpoints/dinov3_full_finetune_best.pth'
    }
    output_dir = 'results'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Store all metrics
    all_metrics = {}
    
    # ==================== RESNET9 EVALUATION ====================
    if os.path.exists(checkpoints['ResNet9']):
        print(f"\n{'='*60}")
        print("EVALUATING RESNET9")
        print(f"{'='*60}")
        
        # Load model and data
        resnet_model = load_resnet9_model(checkpoints['ResNet9'], device)
        resnet_testloader = get_resnet_test_loader(batch_size=128, num_workers=2)
        
        # Evaluate
        resnet_metrics = evaluate_model_with_metrics(
            resnet_model, resnet_testloader, device
        )
        all_metrics['ResNet9'] = resnet_metrics
        
        # Print summary
        print_metrics_summary(resnet_metrics, "ResNet9")
        
        # Plot confusion matrix
        plot_confusion_matrix(
            resnet_metrics['confusion_matrix'],
            CIFAR10_CLASSES,
            'ResNet9 - Confusion Matrix on CIFAR-10',
            f'{output_dir}/resnet9_confusion_matrix.png'
        )
        
        # Plot per-class metrics
        plot_per_class_metrics(
            resnet_metrics,
            CIFAR10_CLASSES,
            'ResNet9',
            f'{output_dir}/resnet9_per_class_metrics.png'
        )
    else:
        print(f"\nWarning: ResNet9 checkpoint not found at {checkpoints['ResNet9']}")
    
    # ==================== VIT CLASSIFIER-ONLY EVALUATION ====================
    if os.path.exists(checkpoints['ViT-Classifier']):
        print(f"\n{'='*60}")
        print("EVALUATING VIT (DINOV3) - CLASSIFIER ONLY")
        print(f"{'='*60}")
        
        # Load model and data
        vit_model = load_vit_model(checkpoints['ViT-Classifier'], device, img_size=448)
        vit_testloader = get_vit_test_loader(batch_size=64, num_workers=2, img_size=448)
        
        # Evaluate
        vit_metrics = evaluate_model_with_metrics(
            vit_model, vit_testloader, device
        )
        all_metrics['ViT-Classifier'] = vit_metrics
        
        # Print summary
        print_metrics_summary(vit_metrics, "ViT (DINOv3) - Classifier Only")
        
        # Plot confusion matrix
        plot_confusion_matrix(
            vit_metrics['confusion_matrix'],
            CIFAR10_CLASSES,
            'ViT (DINOv3) Classifier Only - Confusion Matrix on CIFAR-10',
            f'{output_dir}/vit_confusion_matrix.png'
        )
        
        # Plot per-class metrics
        plot_per_class_metrics(
            vit_metrics,
            CIFAR10_CLASSES,
            'ViT (DINOv3) - Classifier Only',
            f'{output_dir}/vit_per_class_metrics.png'
        )
    else:
        print(f"\nWarning: ViT Classifier checkpoint not found at {checkpoints['ViT-Classifier']}")
    
    # ==================== VIT FULL FINE-TUNING EVALUATION ====================
    if os.path.exists(checkpoints['ViT-FullFT']):
        print(f"\n{'='*60}")
        print("EVALUATING VIT (DINOV3) - FULL FINE-TUNING")
        print(f"{'='*60}")
        
        # Load model and data
        vit_full_model = load_vit_model(checkpoints['ViT-FullFT'], device, img_size=448)
        vit_full_testloader = get_vit_test_loader(batch_size=64, num_workers=2, img_size=448)
        
        # Evaluate
        vit_full_metrics = evaluate_model_with_metrics(
            vit_full_model, vit_full_testloader, device
        )
        all_metrics['ViT-FullFT'] = vit_full_metrics
        
        # Print summary
        print_metrics_summary(vit_full_metrics, "ViT (DINOv3) - Full Fine-Tuning")
        
        # Plot confusion matrix
        plot_confusion_matrix(
            vit_full_metrics['confusion_matrix'],
            CIFAR10_CLASSES,
            'ViT (DINOv3) Full Fine-Tuning - Confusion Matrix on CIFAR-10',
            f'{output_dir}/vit_full_confusion_matrix.png'
        )
        
        # Plot per-class metrics
        plot_per_class_metrics(
            vit_full_metrics,
            CIFAR10_CLASSES,
            'ViT (DINOv3) - Full Fine-Tuning',
            f'{output_dir}/vit_full_per_class_metrics.png'
        )
    else:
        print(f"\nWarning: ViT Full FT checkpoint not found at {checkpoints['ViT-FullFT']}")
    
    # ==================== COMPARISON ====================
    if len(all_metrics) >= 2:
        print(f"\n{'='*60}")
        print("CREATING COMPARISON PLOTS")
        print(f"{'='*60}")
        
        plot_comparison(
            all_metrics,
            CIFAR10_CLASSES,
            f'{output_dir}/model_comparison.png'
        )
        
        # Print comparison summary
        print("\nModel Comparison Summary:")
        model_names = list(all_metrics.keys())
        
        # Header
        header = f"{'Metric':<20}"
        for name in model_names:
            header += f" {name:<18}"
        print(header)
        print("-" * len(header))
        
        # Accuracy
        line = f"{'Accuracy (%)':<20}"
        for name in model_names:
            line += f" {all_metrics[name]['accuracy']:>17.2f} "
        print(line)
        
        # Precision
        line = f"{'Precision':<20}"
        for name in model_names:
            line += f" {all_metrics[name]['precision']:>17.4f} "
        print(line)
        
        # Recall
        line = f"{'Recall':<20}"
        for name in model_names:
            line += f" {all_metrics[name]['recall']:>17.4f} "
        print(line)
        
        # F1 Score
        line = f"{'F1 Score':<20}"
        for name in model_names:
            line += f" {all_metrics[name]['f1_score']:>17.4f} "
        print(line)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"All results saved to '{output_dir}/' directory")
    print("\nGenerated files:")
    if 'ResNet9' in all_metrics:
        print(f"  - {output_dir}/resnet9_confusion_matrix.png")
        print(f"  - {output_dir}/resnet9_per_class_metrics.png")
    if 'ViT-Classifier' in all_metrics:
        print(f"  - {output_dir}/vit_confusion_matrix.png")
        print(f"  - {output_dir}/vit_per_class_metrics.png")
    if 'ViT-FullFT' in all_metrics:
        print(f"  - {output_dir}/vit_full_confusion_matrix.png")
        print(f"  - {output_dir}/vit_full_per_class_metrics.png")
    if len(all_metrics) >= 2:
        print(f"  - {output_dir}/model_comparison.png")


if __name__ == '__main__':
    main()

