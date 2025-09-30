import torch
import torch.nn as nn
from attention import Attention

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout, attention):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size + hidden_size, hidden_size, num_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size * 2 + embedding_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size]
        # hidden, cell : [num_layers, batch_size, hidden_size]
        # encoder_outputs: [src_len, batch_size, hidden_size]

        input = input.unsqueeze(0)  # input: [1, batch_size]

        embedded = self.dropout(self.embedding(input))  # embedded: [1, batch_size, embedding_size]

        # Attention 가중치 계산
        a = self.attention(hidden[-1], encoder_outputs)  # a: [src_len, batch_size]
        a = a.permute(1, 0).unsqueeze(1)  # a: [batch_size, 1, src_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # encoder_outputs: [batch_size, src_len, hidden_size]

        context = torch.bmm(a, encoder_outputs)  # context: [batch_size, 1, hidden_size]

        context = context.permute(1, 0, 2)  # context: [1, batch_size, hidden_size]

        rnn_input = torch.cat((embedded, context), dim=2)  # rnn_input: [1, batch_size, embedding_size + hidden_size]

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs: [1, batch_size, hidden_size]

        # 출력 예측
        prediction = self.fc_out(torch.cat((outputs.squeeze(0), context.squeeze(0), embedded.squeeze(0)), dim=1))
        # prediction: [batch_size, output_size]

        return prediction, hidden, cell