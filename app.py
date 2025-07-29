import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import torch.nn as nn

# 클래스 라벨 정의
labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# CNN 모델 클래스 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# 모델 로드
model = CNN()
model.load_state_dict(torch.load("fashion_mnist_cnn.pth", map_location=torch.device("cpu")))
model.eval()

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 예측 함수
def predict(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        return labels[predicted.item()]

# Gradio 인터페이스 설정
interface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="label", title="👕 FashionMNIST Classifier")

if __name__ == "__main__":
    print("✅ Gradio 웹앱 실행 중... 접속: http://127.0.0.1:7860")
    interface.launch()
