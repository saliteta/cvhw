import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import time
import os

# Try importing the user's custom model, otherwise fallback to ResNet18
try:
    from resnet.net import ResNet9
    HAS_CUSTOM_MODEL = True
except ImportError:
    HAS_CUSTOM_MODEL = False
    print("Custom ResNet9 not found. Falling back to torchvision ResNet18.")

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 0.12     # Max perturbation
ALPHA = 0.006      # Step size
STEPS = 20         # Attack iterations (PGD-20)
BATCH_SIZE = 128   # Batch size for training
TRAIN_EPOCHS = 50  # Number of epochs to train

# --- 1. Normalization Layer ---
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor(std).reshape(1, 3, 1, 1))
        
    def forward(self, x):
        return (x - self.mean) / self.std

# --- 2. Data Loaders ---
def get_cifar10_loaders():
    """
    Returns training and test loaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(), # Keep as [0, 1]
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    # Batch size 128 for faster evaluation
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

# --- 3. Model Loading ---
def load_resnet_model(pretrained=True, checkpoint_path=None):
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2023, 0.1994, 0.2010])

    if HAS_CUSTOM_MODEL:
        base_model = ResNet9(num_classes=10)
        # Load specific checkpoint if provided, else try default best, else random
        if checkpoint_path and os.path.exists(checkpoint_path):
             print(f"Loading weights from {checkpoint_path}")
             checkpoint = torch.load(checkpoint_path, map_location=device)
             # Handle case where checkpoint is full dict or just state_dict
             if 'model_state_dict' in checkpoint:
                 base_model.load_state_dict(checkpoint['model_state_dict'])
             else:
                 base_model.load_state_dict(checkpoint)
        elif pretrained:
            try:
                default_path = 'checkpoints/resnet9_best.pth'
                checkpoint = torch.load(default_path, map_location=device)
                base_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded pretrained ResNet9 from {default_path}")
            except FileNotFoundError:
                print("Pretrained checkpoint not found, using random weights.")
    else:
        base_model = models.resnet18(weights=None)
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()
        base_model.fc = nn.Linear(base_model.fc.in_features, 10)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path}")
            base_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model = nn.Sequential(
        norm_layer,
        base_model
    )
    return model.to(device)

# --- 4. Batched Attack for Training/Eval ---
def igsm_attack_batch(model, images, labels, epsilon=EPSILON, alpha=ALPHA, steps=STEPS):
    """
    Optimized batched IGSM attack.
    """
    images = images.to(device)
    labels = labels.to(device)
    
    # Starting point: Original image + random noise
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1).detach()
    
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        grad = torch.autograd.grad(loss, adv_images)[0]
        
        # Update
        adv_images = adv_images.detach() + alpha * grad.sign()
        
        # Projection (Clipping)
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + delta, 0, 1).detach()
        
    return adv_images

# --- 5. Adversarial Training Loop (Full) ---
def train_adversarial_epochs(model, trainloader, epochs=50, save_dir='./checkpoints'):
    """
    Trains the model on adversarial examples for multiple epochs.
    """
    # Create checkpoint directory
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'resnet_adv.pth')
    
    model.train()
    
    # Optimizer & Scheduler setup
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # Decay LR at 50% and 75% of training
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nStarting Adversarial Training for {epochs} epochs...")
    print(f"Attack: PGD-{STEPS}, Epsilon: {EPSILON}")
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 1. Generate Adversarial Batch
            model.eval() # Eval for generating attack (stops BatchNorm updates)
            adv_inputs = igsm_attack_batch(model, inputs, targets)
            model.train() # Train for updating weights
            
            # 2. Forward pass with ADVERSARIAL images
            optimizer.zero_grad()
            outputs = model(adv_inputs)
            
            # 3. Update
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        # End of epoch stats
        scheduler.step()
        epoch_acc = 100. * correct / total
        epoch_loss = total_loss / len(trainloader)
        duration = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{epochs} | Time: {duration:.0f}s | "
              f"Adv Loss: {epoch_loss:.4f} | Adv Acc: {epoch_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.1e}")
        
        # Save checkpoint periodically and at end
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            # Access the base model (index 1 of Sequential)
            state_to_save = model[1].state_dict()
            torch.save(state_to_save, save_path)
            print(f"  -> Model saved to {save_path}")

    print("Training Complete.")
    return save_path

# --- 6. Evaluation Logic ---
def evaluate_model_robustness(model, dataloader, name="Model"):
    """
    Evaluates clean and adversarial accuracy on the dataset.
    """
    model.eval()
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    print(f"\nEvaluating {name}...")
    start = time.time()
    
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        total += labels.size(0)
        
        # 1. Clean Accuracy
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            clean_correct += predicted.eq(labels).sum().item()
        
        # 2. Adversarial Accuracy
        # Generate attack (requires gradients w.r.t input)
        adv_inputs = igsm_attack_batch(model, inputs, labels)
        
        with torch.no_grad():
            outputs_adv = model(adv_inputs)
            _, predicted_adv = outputs_adv.max(1)
            adv_correct += predicted_adv.eq(labels).sum().item()
            
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(dataloader)} batches...", end='\r')
            
    clean_acc = 100. * clean_correct / total
    adv_acc = 100. * adv_correct / total
    
    print(f"\n{name} Results ({time.time()-start:.1f}s):")
    print(f"  Clean Accuracy:       {clean_acc:.2f}%")
    print(f"  Adversarial Accuracy: {adv_acc:.2f}%")
    
    return clean_acc, adv_acc

# --- 7. Visualization Function ---
def visualize_single_model(model, img, label, classes, save_path, title_prefix):
    """
    Helper to attack a specific model and visualize its performance.
    """
    # 1. Generate Attack against THIS specific model
    adv_img = igsm_attack_batch(model, img, label)
    
    # 2. Get Predictions
    with torch.no_grad():
        clean_logits = model(img)
        adv_logits = model(adv_img)
        clean_probs = F.softmax(clean_logits, dim=1).cpu().numpy()[0]
        adv_probs = F.softmax(adv_logits, dim=1).cpu().numpy()[0]
        adv_pred = adv_logits.max(1)[1].item()
    
    # 3. Plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # Clean Image
    clean_np = img.cpu().squeeze().permute(1, 2, 0).numpy()
    ax1.imshow(clean_np)
    ax1.set_title(f"Clean: {classes[label.item()]}", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Adversarial Image
    adv_np = adv_img.cpu().squeeze().permute(1, 2, 0).numpy()
    ax2.imshow(adv_np)
    title_color = 'red' if adv_pred != label.item() else 'green'
    status = "Fooled" if adv_pred != label.item() else "Resisted"
    ax2.set_title(f"Adversarial (Attacked {title_prefix})\nStatus: {status}\nPrediction: {classes[adv_pred]}", 
                  fontsize=11, color=title_color, fontweight='bold')
    ax2.axis('off')
    
    # Histogram
    x = np.arange(10)
    width = 0.35
    ax3.bar(x - width/2, clean_probs, width, label='Clean Conf', color='skyblue')
    ax3.bar(x + width/2, adv_probs, width, label='Adv Conf', color='salmon')
    
    ax3.set_ylabel('Probability')
    ax3.set_title(f'{title_prefix} Model Confidence Shift')
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=45, ha="right")
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def generate_sample_images(standard_model, robust_model, dataloader, save_dir='adversarial_samples'):
    """
    Generates 20 pairs of images (Standard vs Robust) for the same inputs.
    """
    os.makedirs(save_dir, exist_ok=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    standard_model.eval()
    robust_model.eval()
    
    samples_collected = 0
    
    print(f"\n--- Generating {20} Sample Pairs in '{save_dir}' ---")
    
    for inputs, labels in dataloader:
        if samples_collected >= 20:
            break
            
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        for i in range(inputs.size(0)):
            if samples_collected >= 20:
                break
            
            img = inputs[i:i+1]
            label = labels[i:i+1]
            sample_id = samples_collected + 1
            
            # 1. Standard Model Analysis (sample_X_standard.png)
            save_path_std = os.path.join(save_dir, f"sample_{sample_id:02d}_standard.png")
            visualize_single_model(standard_model, img, label, classes, save_path_std, "Standard")
            
            # 2. Robust Model Analysis (sample_X_robust.png)
            save_path_rob = os.path.join(save_dir, f"sample_{sample_id:02d}_robust.png")
            visualize_single_model(robust_model, img, label, classes, save_path_rob, "Robust")
            
            samples_collected += 1
            
    print(f"Done! Saved {samples_collected * 2} images to directory: {save_dir}")

def main():
    print(f"Setting up Experiment on {device}...")
    
    # 1. Load Data
    trainloader, testloader = get_cifar10_loaders()
    
    # 2. Train the Robust Model
    # Uncomment if you need to train. If you have checkpoints, comment this out.
    print(f"\n--- Phase 1: Training Robust Model ---")
    # model_to_train = load_resnet_model(pretrained=True)
    # saved_path = train_adversarial_epochs(model_to_train, trainloader, epochs=TRAIN_EPOCHS)
    saved_path = './checkpoints/resnet_adv.pth' # Default path
    
    # 3. Evaluation Phase
    print(f"\n--- Phase 2: Final Evaluation ---")
    
    # Load Standard Model
    print("Loading Standard Pretrained Model...")
    standard_model = load_resnet_model(pretrained=True)
    std_clean, std_adv = evaluate_model_robustness(standard_model, testloader, name="Standard Model")
    
    # Load Robust Model
    print("Loading Robust Trained Model...")
    robust_model = load_resnet_model(pretrained=False, checkpoint_path=saved_path)
    rob_clean, rob_adv = evaluate_model_robustness(robust_model, testloader, name="Robust Model")
    
    # 4. Final Comparison Table
    print("\n" + "="*50)
    print(f"{'Metric':<20} | {'Standard Model':<15} | {'Robust Model':<15}")
    print("-" * 55)
    print(f"{'Clean Accuracy':<20} | {std_clean:<14.2f}% | {rob_clean:<14.2f}%")
    print(f"{'Adversarial Acc':<20} | {std_adv:<14.2f}% | {rob_adv:<14.2f}%")
    print("=" * 50)
    
    # 5. Visualize 20 Samples (Standard vs Robust pairs)
    print("\n--- Phase 3: Generating Individual Sample Images ---")
    generate_sample_images(standard_model, robust_model, testloader)

if __name__ == "__main__":
    main()