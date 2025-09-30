import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size

        # 예측 결과를 저장할 텐서
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # Encoder의 마지막 hidden state와 cell state를 Decoder의 초기 hidden state와 cell state로 사용
        encoder_outputs, hidden, cell = self.encoder(src)

        # 디코더의 첫 번째 입력은 항상 <bos> 토큰
        input = trg[0, :]

        for t in range(1, trg_len):
            # 디코더에 현재 입력, 이전 hidden state, cell state, 그리고 encoder outputs를 전달
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)

            # 예측 결과 저장
            outputs[t] = output

            # Teacher Forcing 적용 여부 결정
            teacher_force = random.random() < teacher_forcing_ratio

            # 다음 입력 결정: Teacher Forcing을 사용할 경우 실제 정답 토큰 사용, 그렇지 않으면 예측 토큰 사용
            top1 = output.argmax(1)  # 가장 높은 확률을 가진 단어의 인덱스
            input = trg[t] if teacher_force else top1

        return outputs