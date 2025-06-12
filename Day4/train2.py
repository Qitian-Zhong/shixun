# 完整的模型训练套路(以CIFAR10为例)
import time
import os

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import *
from dataset import ImageTxtDataset

# 设置数据集路径
dataset_root = r"E:\pythonproject\dateset"
train_txt = os.path.join(dataset_root, "train.txt")
val_txt = os.path.join(dataset_root, "val.txt")
train_img_dir = os.path.join(dataset_root, "Images", "train")
val_img_dir = os.path.join(dataset_root, "Images", "val")

# 准备数据集
train_data = ImageTxtDataset(train_txt,
                             train_img_dir,
                             transforms.Compose([transforms.Resize(32),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])
                                                 ])
                             )

test_data = ImageTxtDataset(val_txt,
                           val_img_dir,
                           transforms.Compose([transforms.Resize(32),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])
                                             ])
                           )

# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度{train_data_size}")
print(f"测试数据集的长度{test_data_size}")

# 加载数据集
train_loader = DataLoader(train_data,batch_size=64)
test_loader = DataLoader(test_data,batch_size=64)

# 创建网络模型

chen = Chen()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 1e-2 相当于(10)^(-2)
learning_rate = 0.01
optim = torch.optim.SGD(chen.parameters(),lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0 #记录训练的次数
total_test_step = 0 # 记录测试的次数
epoch = 10 # 训练的轮数

# 添加tensorboard
writer = SummaryWriter("../logs_train")

# 添加开始时间
start_time = time.time()

for i in range(epoch):
    print(f"-----第{i+1}轮训练开始-----")
    # 训练步骤
    for data in train_loader:
        imgs, targets = data
        outputs = chen(imgs)
        loss = loss_fn(outputs,targets)

        # 优化器优化模型
        optim.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optim.step()

        total_train_step += 1
        if total_train_step % 500 == 0:
            print(f"第{total_train_step}的训练的loss:{loss.item()}")
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    end_time = time.time()
    print(f"训练时间{end_time - start_time}")

    # 测试步骤（以测试数据上的正确率来评估模型）
    total_test_loss = 0.0
    # 整体正确个数
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            outputs = chen(imgs)
            # 损失
            loss = loss_fn(outputs,targets)
            total_test_loss += loss.item()
            # 正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的loss:{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy,total_test_step)
    total_test_step += 1

    # 保存每一轮训练模型
    torch.save(chen,f"model_save\\chen_{i}.pth")
    print("模型已保存")

writer.close()