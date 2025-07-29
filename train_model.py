import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def train_model():
    print("✅ 학습 시작")  # 실행 확인용

    # 1. 데이터 로딩 및 전처리
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True
    )

    # 2. CNN 모델 정의
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),  # 28x28 → 28x28
                nn.ReLU(),
                nn.MaxPool2d(2),                # 28x28 → 14x14
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)                 # 14x14 → 7x7
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64*7*7, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            x = self.conv(x)
            x = self.fc(x)
            return x

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. 모델 학습
    for epoch in range(5):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[{epoch+1}] loss: {running_loss/len(trainloader):.4f}")

    # 4. 모델 저장
    torch.save(model.state_dict(), "fashion_mnist_cnn.pth")
    print("✅ 모델 저장 완료: fashion_mnist_cnn.pth")

# 🔧 명시적 실행
if __name__ == "__main__":
    train_model()
