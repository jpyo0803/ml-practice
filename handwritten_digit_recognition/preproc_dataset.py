import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_and_preprocess_mnist(batch_size=64, val_split=0.2):
    # 데이터 Normalize를 위한 파라미터. 평균 = 0.1307, 표준편차 = 0.3081은 미리 계산된 값
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # MNIST 데이터셋 불러오기 
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Train 데이터와 validation 데이터를 8:2로 나누기
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 데이터 로더
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader