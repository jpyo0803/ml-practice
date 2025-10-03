import torch
import torch.nn as nn
import torch.optim as optim
import time
import math

from preproc_dataset import load_and_preprocess_nmt, PAD_IDX
from encoder import Encoder
from attention import Attention
from decoder import Decoder
from seq2seq import Seq2Seq

def train_one_epoch(model, loader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(loader):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        # output: [trg_len, batch_size, output_dim]
        # trg: [trg_len, batch_size]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        # Gradient Clipping을 통해 gradient explosion 방지
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval() # Dropout, BatchNorm 등을 평가 모드로 변경
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(loader):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg, 0) # 평가 시에는 Teacher Forcing 사용 안함

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)

def train(model, loaders, optimizer, criterion, device, num_epoch, clip):
    train_loader, valid_loader, test_loader = loaders

    for epoch in range(num_epoch):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, clip, device)
        valid_loss = validate(model, valid_loader, criterion, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = int((end_time - start_time) / 60), int((end_time - start_time) % 60)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    test_loss = validate(model, test_loader, criterion, device)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    
    return model

if __name__ == "__main__":
    # 데이터셋 로드
    train_loader, val_loader, test_loader, vocabs = load_and_preprocess_nmt(max_vocab_size=15000)

    # 하이퍼파라미터
    INPUT_DIM = len(vocabs['en'])
    OUTPUT_DIM = len(vocabs['de'])
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    NUM_EPOCH = 10
    CLIP = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device (train): ", device)

    # 모델 초기화
    attn = Attention(HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 모델 훈련
    trained_model = train(model, (train_loader, val_loader, test_loader), optimizer, criterion, device, NUM_EPOCH, CLIP)
    
    # 모델 저장
    model_path = "./model_weight.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")