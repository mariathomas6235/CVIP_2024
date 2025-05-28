import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Attention(nn.Module):
    def __init__(self, in_channels, attention_size):
        super().__init__()
        self.query = nn.Conv2d(in_channels, attention_size, 1)
        self.key = nn.Conv2d(in_channels, attention_size, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, C, H * W)
        attn_scores = self.softmax(torch.bmm(q.transpose(1, 2), k) / (H * W) ** 0.5)
        out = torch.bmm(v, attn_scores.transpose(1, 2)).view(B, C, H, W)
        return out + x  

class MobileNetV3WithAttention(nn.Module):
    def __init__(self, num_classes, attention_size=64):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights='DEFAULT')
        in_features = self.backbone.classifier[0].in_features
        self.attn = Attention(in_channels=in_features, attention_size=attention_size)
        self.backbone.classifier = nn.Sequential(nn.Linear(in_features, num_classes))

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.attn(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.backbone.classifier(x)
        return x
