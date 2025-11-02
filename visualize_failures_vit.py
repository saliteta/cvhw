import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from vit.model import create_dinov3_model, get_transform
import os


# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']


def get_vit_test_loader(batch_size=1, num_workers=2, img_size=448, shuffle=False):
    """
    Get CIFAR-10 test data loader for ViT (448x448 images)
    Returns both normalized and unnormalized versions
    """
    # Normalization transform for model
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
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Also get unnormalized version for visualization
    transform_unnorm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    testset_unnorm = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=False, 
        transform=transform_unnorm
    )
    
    testloader_unnorm = DataLoader(
        testset_unnorm, 
        batch_size=batch_size,
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return testloader, testloader_unnorm


def load_vit_model(checkpoint_path, device, img_size=448):
    """
    Load ViT (DINOv3) model from checkpoint
    """
    print(f"Loading ViT (DINOv3) from {checkpoint_path}...")
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


def find_failure_cases(model, testloader, testloader_unnorm, device, num_failures=50):
    """
    Find misclassified examples
    
    Returns:
        List of tuples: (image, true_label, predicted_label, confidence, probabilities)
    """
    print("\nFinding failure cases...")
    model.eval()
    
    failures = []
    
    with torch.no_grad():
        for (inputs, targets), (inputs_unnorm, _) in zip(testloader, testloader_unnorm):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
            
            # Check if misclassified
            if predicted.item() != targets.item():
                failures.append({
                    'image': inputs_unnorm[0].cpu().numpy().transpose(1, 2, 0),
                    'true_label': targets.item(),
                    'predicted_label': predicted.item(),
                    'confidence': confidence.item(),
                    'probabilities': probabilities[0].cpu().numpy()
                })
                
                if len(failures) >= num_failures:
                    break
    
    print(f"Found {len(failures)} failure cases")
    return failures


def visualize_failure_cases(failures, num_examples=5, save_path='results/vit_failures.png'):
    """
    Visualize failure cases in a grid
    """
    if len(failures) == 0:
        print("No failure cases to visualize!")
        return
    
    if num_examples > len(failures):
        num_examples = len(failures)
    
    # Select examples to show
    selected_failures = failures[:num_examples]
    
    # Create figure
    fig, axes = plt.subplots(1, num_examples, figsize=(4 * num_examples, 5))
    
    if num_examples == 1:
        axes = [axes]
    
    for idx, (ax, failure) in enumerate(zip(axes, selected_failures)):
        # Display image
        img = failure['image']
        ax.imshow(img)
        ax.axis('off')
        
        # Add title with true and predicted labels
        true_class = CIFAR10_CLASSES[failure['true_label']]
        pred_class = CIFAR10_CLASSES[failure['predicted_label']]
        confidence = failure['confidence']
        
        title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2%}"
        ax.set_title(title, fontsize=12, pad=10)
        
        # Add border color (red for wrong)
        for spine in ax.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
    
    plt.suptitle('ViT (DINOv3) Failure Cases on CIFAR-10', fontsize=16, y=0.98)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved failure cases visualization to {save_path}")
    plt.close()


def visualize_detailed_failures(failures, num_examples=5, save_path='results/vit_failures_detailed.png'):
    """
    Visualize failure cases with top-3 predictions
    """
    if len(failures) == 0:
        print("No failure cases to visualize!")
        return
    
    if num_examples > len(failures):
        num_examples = len(failures)
    
    selected_failures = failures[:num_examples]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 4 * num_examples))
    
    for idx, failure in enumerate(selected_failures):
        # Image subplot
        ax_img = plt.subplot(num_examples, 4, idx * 4 + 1)
        img = failure['image']
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title(f"Sample {idx + 1}", fontsize=12, fontweight='bold')
        
        # Add red border
        for spine in ax_img.spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(3)
        
        # True label info
        ax_true = plt.subplot(num_examples, 4, idx * 4 + 2)
        ax_true.text(0.5, 0.5, f"TRUE LABEL:\n{CIFAR10_CLASSES[failure['true_label']].upper()}", 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax_true.axis('off')
        
        # Predicted label info
        ax_pred = plt.subplot(num_examples, 4, idx * 4 + 3)
        ax_pred.text(0.5, 0.5, f"PREDICTED:\n{CIFAR10_CLASSES[failure['predicted_label']].upper()}\n({failure['confidence']:.1%})", 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax_pred.axis('off')
        
        # Top-3 predictions bar chart
        ax_bar = plt.subplot(num_examples, 4, idx * 4 + 4)
        probs = failure['probabilities']
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3_probs = probs[top3_indices]
        top3_labels = [CIFAR10_CLASSES[i] for i in top3_indices]
        
        colors = ['red' if i == failure['predicted_label'] else 'skyblue' for i in top3_indices]
        bars = ax_bar.barh(range(3), top3_probs, color=colors, alpha=0.7)
        ax_bar.set_yticks(range(3))
        ax_bar.set_yticklabels(top3_labels)
        ax_bar.set_xlabel('Confidence', fontsize=10)
        ax_bar.set_title('Top-3 Predictions', fontsize=11)
        ax_bar.set_xlim(0, 1.0)
        ax_bar.grid(axis='x', alpha=0.3)
        
        # Add percentage labels on bars
        for i, (bar, prob) in enumerate(zip(bars, top3_probs)):
            ax_bar.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontsize=9)
    
    plt.suptitle('ViT (DINOv3) Detailed Failure Analysis on CIFAR-10', fontsize=18, y=0.995)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved detailed failure analysis to {save_path}")
    plt.close()


def analyze_failure_patterns(failures):
    """
    Analyze common patterns in failures
    """
    print("\n" + "="*60)
    print("FAILURE PATTERN ANALYSIS")
    print("="*60)
    
    if len(failures) == 0:
        print("\nNo failures found! Model has perfect accuracy on test set.")
        print("="*60 + "\n")
        return
    
    # Count failures by true class
    true_class_counts = {}
    pred_class_counts = {}
    confusion_pairs = {}
    
    for failure in failures:
        true_label = failure['true_label']
        pred_label = failure['predicted_label']
        
        true_class_counts[true_label] = true_class_counts.get(true_label, 0) + 1
        pred_class_counts[pred_label] = pred_class_counts.get(pred_label, 0) + 1
        
        pair = (true_label, pred_label)
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    # Most commonly misclassified true classes
    print("\nMost commonly misclassified TRUE classes:")
    sorted_true = sorted(true_class_counts.items(), key=lambda x: x[1], reverse=True)
    for class_idx, count in sorted_true[:5]:
        print(f"  {CIFAR10_CLASSES[class_idx]:<12} : {count:>3} failures")
    
    # Most common wrong predictions
    print("\nMost common WRONG predictions:")
    sorted_pred = sorted(pred_class_counts.items(), key=lambda x: x[1], reverse=True)
    for class_idx, count in sorted_pred[:5]:
        print(f"  {CIFAR10_CLASSES[class_idx]:<12} : {count:>3} times")
    
    # Most common confusion pairs
    print("\nMost common confusion pairs (True → Predicted):")
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    for (true_idx, pred_idx), count in sorted_pairs[:min(10, len(sorted_pairs))]:
        print(f"  {CIFAR10_CLASSES[true_idx]:<12} → {CIFAR10_CLASSES[pred_idx]:<12} : {count:>3} times")
    
    # Average confidence on failures
    avg_confidence = np.mean([f['confidence'] for f in failures])
    print(f"\nAverage confidence on failures: {avg_confidence:.2%}")
    print("="*60 + "\n")


def main():
    # Configuration
    checkpoint_path = 'checkpoints/dinov3_best.pth'
    output_dir = 'results'
    num_failures_to_find = 100  # ViT has very few failures, so we search more
    num_to_visualize = 5
    img_size = 448
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    model = load_vit_model(checkpoint_path, device, img_size=img_size)
    
    # Load data
    print("\nLoading CIFAR-10 test set...")
    testloader, testloader_unnorm = get_vit_test_loader(
        batch_size=1, num_workers=2, img_size=img_size, shuffle=False
    )
    
    # Find failure cases
    failures = find_failure_cases(
        model, testloader, testloader_unnorm, device, 
        num_failures=num_failures_to_find
    )
    
    if len(failures) == 0:
        print("\n" + "="*60)
        print("🎉 NO FAILURE CASES FOUND!")
        print("="*60)
        print("The ViT model achieved 100% accuracy on the test set!")
        print("This is extremely impressive performance.")
        return
    
    # Analyze patterns
    analyze_failure_patterns(failures)
    
    # Visualize failures (if any)
    if len(failures) > 0:
        print(f"\nGenerating visualizations...")
        visualize_failure_cases(
            failures, 
            num_examples=min(num_to_visualize, len(failures)),
            save_path=f'{output_dir}/vit_failures.png'
        )
        
        visualize_detailed_failures(
            failures,
            num_examples=min(num_to_visualize, len(failures)),
            save_path=f'{output_dir}/vit_failures_detailed.png'
        )
        
        print(f"\n{'='*60}")
        print("VISUALIZATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Generated files:")
        print(f"  - {output_dir}/vit_failures.png")
        print(f"  - {output_dir}/vit_failures_detailed.png")
        print(f"Total failure cases analyzed: {len(failures)}")


if __name__ == '__main__':
    main()

