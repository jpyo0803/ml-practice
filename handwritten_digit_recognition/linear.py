import torch 
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features)) 

    def forward(self, x):
        # 만약 x가 batch 형태라면 torch.matmul이 자동으로 브로드캐스팅을 처리 
        return torch.matmul(x, self.weights) + self.bias
