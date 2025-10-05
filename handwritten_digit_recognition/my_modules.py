import torch.nn as nn

from relu import my_relu
from linear import MyLinear
from conv2d import MyConv2d

# Convolution + BatchNorm + Relu 모듈 
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() # nn.Module을 상속받으면 필수적으로 불러주어야 함
        self.conv = MyConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # (Conv) -> (Batch Norm) -> (Relu)
        return my_relu(self.bn(self.conv(x)))

# Fully Connected Layer 모듈 
class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super().__init__()
        self.fc = MyLinear(in_features, out_features)
        # dropout은 base 아키텍처를 기반으로 변종 아키텍처를 두는 효과를 통해 overfitting을 완화
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(my_relu(self.fc(x)))