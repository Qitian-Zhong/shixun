import torch
from torch import nn


class alex(nn.Module):
    def __init__(self):
        super(alex, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),  # 32x32
            nn.MaxPool2d(kernel_size=2),  # 16x16
            nn.Conv2d(48, 128, kernel_size=3, padding=1),  # 16x16
            nn.MaxPool2d(kernel_size=2),  # 8x8
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # 8x8
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # 8x8
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # 8x8
            nn.MaxPool2d(kernel_size=2),  # 4x4
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)  # CIFAR10的输入尺寸
    alexnet = alex()
    y = alexnet(x)
    print(y.shape)
