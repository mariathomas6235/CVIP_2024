import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .attention import attention

class mobilenetv3(nn.Module):
    def __init__(self, num_classes=10, attention_size=64):
        super(mobilenetv3, self).__init__()
        self.mobilenet_v3 = models.mobilenet_v3_large(weights='DEFAULT')
        in_features = self.mobilenet_v3.classifier[0].in_features
        self.self_attention = attention(in_channels=in_features, attention_size=attention_size)
        self.mobilenet_v3.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.mobilenet_v3.features(x)
        x = self.self_attention(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.mobilenet_v3.classifier(x)
        return x
