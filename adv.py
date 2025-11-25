import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Try importing the user's custom model, otherwise fallback to ResNet18
try:
    from resnet.net import ResNet9
    HAS_CUSTOM_MODEL = True
except ImportError:
    HAS_CUSTOM_MODEL = False
    print("Custom ResNet9 not found. Falling back to torchvision ResNet18.")

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 0.12    # Max perturbation (Now in [0,1] pixel space)
ALPHA = 0.006     # Step size per iteration
STEPS = 20        # Number of iterations
BATCH_SIZE = 1

# --- 1. Normalization Layer ---
# We define normalization as a layer so we can attack raw [0,1] images
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor(std).reshape(1, 3, 1, 1))
        
    def forward(self, x):
        return (x - self.mean) / self.std

def get_cifar10_data():
    """
    Downloads and loads CIFAR-10 data.
    CRITICAL CHANGE: We do NOT normalize here. We want raw [0, 1] images
    so the attack math (epsilon, clipping) works intuitively.
    """
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts to [0, 1]
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=True)
    return testloader

def load_resnet_model():
    """
    Loads the model and wraps it with the Normalization layer.
    """
    # 1. Define the specific normalization for CIFAR-10
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], 
                           std=[0.2023, 0.1994, 0.2010])

    if HAS_CUSTOM_MODEL:
        base_model = ResNet9(num_classes=10)
        try:
            checkpoint_path = 'checkpoints/resnet9_best.pth'
            checkpoint = torch.load(checkpoint_path, map_location=device)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded trained ResNet9 from {checkpoint_path}")
        except FileNotFoundError:
            print("Checkpoint not found, using random weights for ResNet9.")
    else:
        # Fallback to ResNet18 adapted for CIFAR-10
        base_model = models.resnet18(weights=None)
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base_model.maxpool = nn.Identity()
        base_model.fc = nn.Linear(base_model.fc.in_features, 10)

    # 2. Wrap the model: Input [0,1] -> Normalize -> ResNet
    model = nn.Sequential(
        norm_layer,
        base_model
    )
    
    model = model.to(device)
    model.eval() 
    return model

def igsm_attack_with_history(model, image, label, epsilon, alpha, steps):
    """
    Performs IGSM and returns intermediate steps with full probability distributions.
    Input image should be in range [0, 1].
    """
    # Create a copy of the image to perturb
    adv_image = image.clone().detach().to(device)
    original_image = image.clone().detach().to(device)
    
    history = []
    
    # Helper to capture state
    def capture_state(step_num, img):
        with torch.no_grad():
            output = model(img)
            # Get full probability distribution
            probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
            # Get top prediction info
            pred = probs.argmax()
            pred_prob = probs[pred]
            return (step_num, img.clone().cpu(), pred, pred_prob, probs)

    # Store initial state
    history.append(capture_state(0, adv_image))
    
    for i in range(steps):
        adv_image.requires_grad = True
        
        outputs = model(adv_image)
        loss = nn.CrossEntropyLoss()(outputs, label)
        
        model.zero_grad()
        loss.backward()
        
        data_grad = adv_image.grad.data
        sign_data_grad = data_grad.sign()
        
        perturbed_image = adv_image + alpha * sign_data_grad
        
        # Clip 1: Epsilon constraint
        eta = torch.clamp(perturbed_image - original_image, min=-epsilon, max=epsilon)
        adv_image = original_image + eta
        
        # Clip 2: Valid image range [0, 1]
        adv_image = torch.clamp(adv_image, 0, 1).detach()
        
        # Save snapshot every few steps (or if it succeeds)
        if (i + 1) % 5 == 0 or (i + 1) == steps:
            history.append(capture_state(i + 1, adv_image))
                
        print(f"Step {i+1}/{steps} Loss: {loss.item():.4f}")

    return adv_image, history

def show_attack_evolution(history, classes, true_label_idx):
    """
    Visualizes the progression of the attack over steps with probability bars.
    """
    num_snaps = len(history)
    # Create 2 rows: Top for Images, Bottom for Probability Bars
    fig, axes = plt.subplots(2, num_snaps, figsize=(3 * num_snaps, 8))
    
    # Ensure axes is 2D array even if num_snaps=1
    if num_snaps == 1:
        axes = np.array([axes]).T 
    
    true_label_name = classes[true_label_idx]
    
    for i, (step, img_tensor, pred_idx, prob, full_probs) in enumerate(history):
        # --- Top Row: Image ---
        ax_img = axes[0, i]
        img_np = img_tensor.squeeze().numpy().transpose(1, 2, 0)
        
        pred_name = classes[pred_idx]
        color = 'green' if pred_idx == true_label_idx else 'red'
        
        ax_img.imshow(img_np)
        ax_img.set_title(f"Step {step}\nPred: {pred_name}\nConf: {prob:.2f}", 
                         color=color, fontsize=11, fontweight='bold')
        ax_img.axis('off')

        # --- Bottom Row: Bar Chart ---
        ax_bar = axes[1, i]
        
        # Color logic: Green for True Class, Red for Wrong Prediction, Blue for others
        bar_colors = []
        for j in range(len(classes)):
            if j == true_label_idx:
                bar_colors.append('green') # True Label
            elif j == pred_idx:
                bar_colors.append('red')   # Wrong Prediction
            else:
                bar_colors.append('lightgray')

        ax_bar.bar(range(len(classes)), full_probs, color=bar_colors)
        ax_bar.set_ylim([0, 1.05]) # Fix y-axis to 0-1 range
        
        # Only show y-axis labels on the first plot to reduce clutter
        if i == 0:
            ax_bar.set_ylabel("Probability")
        else:
            ax_bar.set_yticks([])
            
        # X-axis labels
        ax_bar.set_xticks(range(len(classes)))
        ax_bar.set_xticklabels(classes, rotation=90, fontsize=9)
        ax_bar.grid(axis='y', linestyle='--', alpha=0.3)
        
    plt.suptitle(f"IGSM Attack Evolution (Target: {true_label_name})", fontsize=16)
    plt.tight_layout()
    plt.savefig('attack_evolution_dist.png')
    print("Evolution plot saved to 'attack_evolution_dist.png'")
    plt.show()

def main():
    print(f"Running IGSM Attack on {device}...")
    
    model = load_resnet_model()
    loader = get_cifar10_data()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Get image
    dataiter = iter(loader)
    image, label = next(dataiter)
    image, label = image.to(device), label.to(device)

    print(f"True Label: {classes[label.item()]}")

    # Run Attack with History
    adv_image, history = igsm_attack_with_history(model, image, label, EPSILON, ALPHA, STEPS)
    
    # Visualize Steps
    show_attack_evolution(history, classes, label.item())

if __name__ == "__main__":
    main()