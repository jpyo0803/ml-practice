import torch
import torch.nn as nn
import torch.optim as optim

from preproc_dataset import load_and_preprocess_mnist
from simple_cnn import SimpleCNN

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() # gradient를 0으로 초기화해 잘못된 누적 방지 
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward() # 역전파 진행 
        optimizer.step() # 가중치 갱신

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss/total, correct/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            # Dataset을 타겟 디바이스로 이동 
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
        return running_loss/total, correct/total

def train(model, loaders, criterion, optimizer, device, num_epoch):
    train_loader, val_loader, test_loader = loaders

    # 학습 시작전 정확도 확인 
    init_loss, init_acc = validate(model, val_loader, criterion, device)
    print(f"Before Training | Val loss: {init_loss:.4f} acc: {init_acc:.4f}")

    # 학습 
    for epoch in range(num_epoch):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch+1}] Train loss: {train_loss:.4f} acc: {train_acc:.4f} | Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

    # 학습된 모델 Evaluation
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# 여기서 학습 

# 데이터셋 로드
train_loader, val_loader, test_loader = load_and_preprocess_mnist() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device (train): ", device)
model = SimpleCNN().to(device)

num_epoch = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train(model, (train_loader, val_loader, test_loader), criterion, optimizer, device, num_epoch)







