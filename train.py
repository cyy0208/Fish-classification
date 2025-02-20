import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import torchvision
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb
from thop import profile

# 初始化设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 训练轮次和学习率策略
EPOCHS = 30
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 模型初始化
model = torchvision.models.vgg11(pretrained=False).to(device)

# 加载训练和测试数据的 DataLoader
# train_loader, test_loader, df_train_log, df_test_log 初始化

# 用于日志记录和评估的函数
def train_one_batch(images, labels, model, optimizer, epoch, batch_idx):
    """
    运行一个 batch 的训练，返回当前 batch 的训练日志
    """
    images, labels = images.to(device), labels.to(device)
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    
    log_train = {
        'epoch': epoch,
        'batch': batch_idx,
        'train_loss': loss,
        'train_accuracy': accuracy_score(labels.cpu().numpy(), preds)
    }
    
    return log_train

def evaluate_testset(model, test_loader):
    """
    在整个测试集上评估，返回分类评估指标日志
    """
    model.eval()
    loss_list, labels_list, preds_list = [], [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss_list.append(loss.item())
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())
    
    log_test = {
        'test_loss': np.mean(loss_list),
        'test_accuracy': accuracy_score(labels_list, preds_list),
        'test_precision': precision_score(labels_list, preds_list, average='macro'),
        'test_recall': recall_score(labels_list, preds_list, average='macro'),
        'test_f1-score': f1_score(labels_list, preds_list, average='macro')
    }
    
    return log_test

# 定义训练过程
def train_model(train_loader, test_loader, model, optimizer, EPOCHS):
    """
    训练模型并记录日志
    """
    wandb.init(project='new_fish', name=time.strftime('%m%d%H%M%S'))
    df_train_log = pd.DataFrame()
    df_test_log = pd.DataFrame()

    best_test_accuracy = 0.0

    for epoch in range(1, EPOCHS+1):
        print(f'Epoch {epoch}/{EPOCHS}')
        
        # 训练阶段
        model.train()
        for batch_idx, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            log_train = train_one_batch(images, labels, model, optimizer, epoch, batch_idx)
            df_train_log = df_train_log.append(log_train, ignore_index=True)
            wandb.log(log_train)
        
        lr_scheduler.step()

        # 测试阶段
        log_test = evaluate_testset(model, test_loader)
        df_test_log = df_test_log.append(log_test, ignore_index=True)
        wandb.log(log_test)

        # 保存最佳模型
        if log_test['test_accuracy'] > best_test_accuracy:
            best_test_accuracy = log_test['test_accuracy']
            new_best_checkpoint_path = f'checkpoints/best-{best_test_accuracy:.3f}.pth'
            torch.save(model, new_best_checkpoint_path)
            print(f'Saved best model: {new_best_checkpoint_path}')
    
    return df_train_log, df_test_log

# 训练模型
df_train_log, df_test_log = train_model(train_loader, test_loader, model, optimizer, EPOCHS)

# 保存训练日志
df_train_log.to_csv('train_log.csv', index=False)
df_test_log.to_csv('test_log.csv', index=False)

# 打印最佳测试集评估结果
model = torch.load(f'checkpoints/best-{best_test_accuracy:.3f}.pth').eval()
print(evaluate_testset(model, test_loader))

# 打印模型的 FLOPS 和参数数量
dummy_input = torch.randn(2, 3, 224, 224)
flops, params = profile(model, (dummy_input,))
print(f'FLOPS: {flops/1e6:.2f} M, Parameters: {params/1e6:.2f} M')
