import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch_size, hidden_size]
        # encoder_outputs: [src_len, batch_size, hidden_size]

        '''
            hidden는 디코더의 현재 hidden state (t-1), Query, shape: [1, batch_size, hidden_size]
            encoder_outputs는 인코더의 모든 hidden states, Values, shape: [src_len, batch_size, hidden_size]
        '''

        src_len = encoder_outputs.shape[0] # sequence length

        # hidden를 src_len만큼 반복해 encoder outputs와 concat할 수 있게 만듦
        hidden = hidden.repeat(src_len, 1, 1) # [src_len, batch_size, hidden_size]

        '''
            After concat shape: [src_len, batch_size, hidden_size * 2]
            After tanh and linear layer: [src_len, batch_size, hidden_size]
            After final linear layer and squeeze: [src_len, batch_size]

            energy = v^T * tanh(W * [hidden; encoder_outputs])
        '''
        energy = torch.tanh(self.w(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [src_len, batch_size, hidden_size]
        attention = self.v(energy).squeeze(2)
        # attention: [src_len, batch_size]
        return F.softmax(attention, dim=0)  # src_len을 기준으로 소프트맥스 적용