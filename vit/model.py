import timm
import torch
import torch.nn as nn
import torchvision.transforms as T


class DinoV3Classifier(nn.Module):
    """
    DINOv3 Vision Transformer with classifier head using timm
    """
    def __init__(
        self, 
        num_classes=10,
        model_name='vit_large_patch16_dinov3',
        pretrained=True,
        freeze_backbone=False,
        img_size=448,
        pretrained_path=None
    ):
        """
        Args:
            num_classes (int): Number of output classes
            model_name (str): timm model name
            pretrained (bool): Load pretrained weights
            freeze_backbone (bool): Freeze backbone weights
            img_size (int): Input image size (default: 448)
            pretrained_path (str): Path to local pretrained weights file
        """
        super(DinoV3Classifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.img_size = img_size
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained DINOv3 model from timm or local file
        if pretrained_path:
            print(f"Loading {model_name} with local weights from {pretrained_path}...")
            self.backbone = timm.create_model(
                model_name, 
                pretrained=False,
                num_classes=0,  # Remove classification head, get features only
                pretrained_cfg_overlay=dict(file=pretrained_path)
            )
            # Load the weights manually if needed
            if pretrained:
                print(f"Loading pretrained weights from {pretrained_path}...")
                state_dict = torch.load(pretrained_path, map_location='cpu')
                # Handle different checkpoint formats
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                self.backbone.load_state_dict(state_dict, strict=False)
        else:
            print(f"Loading {model_name} from timm...")
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained,
                num_classes=0  # Remove classification head, get features only
            )
        
        # Get feature dimension
        self.embed_dim = self.backbone.num_features
        print(f"Backbone embedding dimension: {self.embed_dim}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            print("Freezing backbone weights...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )
        
        # Initialize classifier
        self._init_classifier()
        
    def _init_classifier(self):
        """Initialize the classifier head"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (batch_size, 3, height, width)
        Returns:
            Output logits (batch_size, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)  # (batch_size, embed_dim)
        
        # Classify
        logits = self.classifier(features)  # (batch_size, num_classes)
        
        return logits
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone weights"""
        print("Unfreezing backbone weights...")
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
    
    def get_num_params(self):
        """Get number of trainable and total parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params, total_params


def get_transform(img_size=448, is_training=True):
    """
    Get preprocessing transforms for DINOv3
    
    Args:
        img_size (int): Target image size
        is_training (bool): Whether for training (applies augmentation)
    
    Returns:
        torchvision.transforms.Compose
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_training:
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    
    return transform


def create_dinov3_model(
    num_classes=10,
    model_name='vit_large_patch16_dinov3',
    pretrained=True,
    freeze_backbone=False,
    img_size=448,
    pretrained_path=None
):
    """
    Create DINOv3 classifier model
    
    Args:
        num_classes (int): Number of output classes
        model_name (str): timm model name
        pretrained (bool): Load pretrained weights
        freeze_backbone (bool): Freeze backbone weights
        img_size (int): Input image size
    
    Returns:
        DinoV3Classifier model
    """
    model = DinoV3Classifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        img_size=img_size,
        pretrained_path=pretrained_path
    )
    
    trainable, total = model.get_num_params()
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters: {total - trainable:,}")
    
    return model

