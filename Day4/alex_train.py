
import time
import os
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import ImageTxtDataset
from alex import MyAlexNet  # 自定义模型导入
from torchvision import transforms
import numpy as np
from datetime import datetime


# 自定义计时器装饰器
def training_timer(func):
    """记录函数执行时间的装饰器"""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"【{func.__name__}】执行耗时: {end - start:.2f}秒")
        return result

    return wrapper


# 自定义数据增强配置
def create_augmentation_pipeline():
    """创建个性化数据增强流水线"""
    return transforms.Compose([
        transforms.Resize((256, 256)),  # 统一图像尺寸
        transforms.RandomHorizontalFlip(p=0.6),  # 增加随机性
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色调整
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
    ])


# 自定义学习率调度器
class CustomLRScheduler:
    """个性化学习率调度器"""

    def __init__(self, optimizer, base_lr, decay_factor=0.95, step_size=3):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.decay_factor = decay_factor
        self.step_size = step_size
        self.epoch_count = 0

    def step(self):
        """更新学习率"""
        self.epoch_count += 1
        if self.epoch_count % self.step_size == 0:
            new_lr = self.base_lr * (self.decay_factor ** (self.epoch_count // self.step_size))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"学习率更新: {new_lr:.6f}")


# 自定义模型评估函数
@training_timer
def evaluate_model(model, data_loader, loss_fn, device):
    """模型评估函数"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据增强配置
    augmentation = create_augmentation_pipeline()

    # 创建数据集
    train_dataset = ImageTxtDataset(
        r"D:\Python\Day3\dataset\train.txt",
        r"D:\Python\Day3\dataset\train.txt",
        transform=augmentation
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./dataset_alex",
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    # 数据集信息
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    print(f"训练集样本数: {train_size}, 测试集样本数: {test_size}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始化模型
    model = MyAlexNet().to(device)
    print("模型架构:")
    print(model)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器配置
    base_lr = 0.01
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = CustomLRScheduler(optimizer, base_lr)

    # 训练参数
    num_epochs = 10
    log_interval = 500
    model_save_dir = "model_checkpoints"
    os.makedirs(model_save_dir, exist_ok=True)

    # 创建TensorBoard日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./training_logs_{timestamp}"
    writer = SummaryWriter(log_dir)

    # 训练循环
    print(f"\n{'=' * 40} 训练开始 {'=' * 40}")
    global_step = 0
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        processed_samples = 0

        print(f"\n▶▶ 第 {epoch + 1}/{num_epochs} 轮训练 ◀◀")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计信息
            running_loss += loss.item()
            processed_samples += labels.size(0)
            global_step += 1

            # 定期日志记录
            if global_step % log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                print(f"批次: {batch_idx + 1}/{len(train_loader)}, "
                      f"损失: {avg_loss:.4f}, "
                      f"已处理: {processed_samples}/{train_size}")
                writer.add_scalar("训练/批次损失", loss.item(), global_step)

        # 更新学习率
        lr_scheduler.step()

        # 评估模型
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"测试结果 - 损失: {test_loss:.4f}, 准确率: {test_acc:.4f}")

        # TensorBoard记录
        writer.add_scalar("训练/轮次损失", running_loss / len(train_loader), epoch)
        writer.add_scalar("测试/损失", test_loss, epoch)
        writer.add_scalar("测试/准确率", test_acc, epoch)

        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model_path = os.path.join(model_save_dir, f"best_model_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': test_loss,
                'accuracy': test_acc
            }, best_model_path)
            print(f"★ 保存最佳模型 (准确率: {test_acc:.4f})")

        # 保存当前模型
        current_model_path = os.path.join(model_save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), current_model_path)
        print(f"保存模型至: {current_model_path}")

        epoch_time = time.time() - epoch_start
        print(f"本轮耗时: {epoch_time:.2f}秒")

    # 训练结束
    writer.close()
    total_time = time.time() - start_time
    print(f"\n{'=' * 40} 训练完成 {'=' * 40}")
    print(f"总耗时: {total_time:.2f}秒, 最佳准确率: {best_accuracy:.4f}")
    print(f"日志目录: {log_dir}")
    print(f"模型保存至: {model_save_dir}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"脚本总运行时间: {time.time() - start_time:.2f}秒")