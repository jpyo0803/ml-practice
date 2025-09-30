import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch_size, hidden_size]
        # encoder_outputs: [src_len, batch_size, hidden_size]

        src_len = encoder_outputs.shape[0] # 입력의 시퀀스 길이

        # hidden를 src_len만큼 반복해 encoder outputs와 concat할 수 있게 만듦
        hidden = hidden.repeat(src_len, 1, 1) # [src_len, batch_size, hidden_size]

        # concat 후 에너지 계산
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [src_len, batch_size, hidden_size]
        attention = self.v(energy).squeeze(2)
        # attention: [src_len, batch_size]
        return F.softmax(attention, dim=0)  # src_len을 기준으로 소프트맥스 적용