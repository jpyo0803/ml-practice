import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        '''
            input_size: source vocab의 크기 
            embedding_size: embedding dimension
        '''

        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding은 토큰 인덱스를 embedding vector로 변환
        self.embedding = nn.Embedding(input_size, embedding_size)
        '''
            시퀀스 앞뒤로 문맥을 유지하기 위해 bidirectional LSTM 사용
        '''
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        '''
            src: [sequence len, batch size]
            embedded: [sequence len, batch size, embedding size]
            outputs: [sequence len, batch size, hidden size * 2]
            hidden: [num layers * 2, batch size, hidden size]
            cell: [num layers * 2, batch size, hidden size]

            hidden, cell은 양방향이라서 2배가 됨
        '''
        # src: [sequence len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded: [sequence len, batch size, embedding size]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [sequence len, batch size, hidden size * 2]
        outputs = torch.tanh(self.fc(outputs))
        return outputs, hidden, cell