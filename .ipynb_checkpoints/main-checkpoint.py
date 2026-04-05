import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Стандартная нормализация для MNIST
])

train_full_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"Размер обучающей выборки: {len(train_full_dataset)}")  # Ожидается 60000
print(f"Размер тестовой выборки: {len(test_dataset)}")          # Ожидается 10000