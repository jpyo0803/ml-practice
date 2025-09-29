import torch
import torch.nn as nn

from preproc_dataset import load_and_preprocess_mnist
from simple_cnn import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader = load_and_preprocess_mnist()

model = SimpleCNN().to(device)

model_path = "./weight/model_weight.pth"
state_dict = torch.load(model_path)

model.load_state_dict(state_dict)

model.eval()

correct = total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        preds = outputs.argmax(dim=1)

        for i in range(len(imgs)):
            if preds[i].item() == labels[i].item():
                correct += 1
            total += 1
            print(f"Predicted: {preds[i].item()}, Ground Truth: {labels[i].item()}, Correct: {preds[i].item() == labels[i].item()}")

print(f"Accuracy: {correct / total:.4f}")