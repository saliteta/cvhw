import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet9(nn.Module):
    """
    ResNet-9 architecture
    A lightweight ResNet with 9 layers total
    """
    def __init__(self, num_classes=10):
        super(ResNet9, self).__init__()
        self.layer1 = Layer1()
        self.layer2 = Layer2()
        self.layer3 = Layer3()
        self.layer4 = Layer4()
        self.classifier = Classifier(num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x


class ConvBlock(nn.Module):
    """
    A convolutional block with batch normalization and ReLU activation
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Layer1(nn.Module):
    """
    A layer of the ResNet-9 architecture
    """
    def __init__(self):
        super(Layer1, self).__init__()

        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        return x


class Layer2(nn.Module):
    """
    A layer of the ResNet-9 architecture
    """
    def __init__(self):
        super(Layer2, self).__init__()
        self.conv1 = ConvBlock(128, 128)
        self.conv2 = ConvBlock(128, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + skip
        x = self.maxpool(x)
        return x

class Layer3(nn.Module):
    """
    A layer of the ResNet-9 architecture
    """
    def __init__(self):
        super(Layer3, self).__init__()
        self.conv1 = ConvBlock(128, 256)
        self.conv2 = ConvBlock(256, 512)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.skip_connection = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512)
        )
    
    def forward(self, x):
        skip = self.skip_connection(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + skip
        x = self.maxpool(x)
        return x

class Layer4(nn.Module):
    """
    A layer of the ResNet-9 architecture
    """
    def __init__(self):
        super(Layer4, self).__init__()
        self.conv1 = ConvBlock(512, 512)
        self.conv2 = ConvBlock(512, 512)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    
    def forward(self, x):
        skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + skip
        x = self.maxpool(x)
        return x


class Classifier(nn.Module):
    """
    A classifier for the ResNet-9 architecture
    """
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x