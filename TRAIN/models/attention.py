import torch
import torch.nn as nn

class attention(nn.Module):
    def __init__(self, in_channels, attention_size):
        super(attention, self).__init__()
        self.query = nn.Conv2d(in_channels, attention_size, kernel_size=1)
        self.key = nn.Conv2d(in_channels, attention_size, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, c, h, w = x.size()
        query = self.query(x).view(batch_size, -1, h * w)
        key = self.key(x).view(batch_size, -1, h * w)
        value = self.value(x).view(batch_size, c, h * w)
        attention_scores = torch.bmm(query.transpose(1, 2), key)
        attention_scores = attention_scores / (h * w) ** 0.5
        attention_map = self.softmax(attention_scores)
        out = torch.bmm(value, attention_map.transpose(1, 2))
        out = out.view(batch_size, c, h, w)
        return out + x
