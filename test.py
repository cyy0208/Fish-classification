import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# 测试集图像预处理：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集文件夹路径
dataset_dir = '/media/zy/600b4f4d-dc6c-4b45-8c58-476ec9afb3fb/cyy/work/图像分类/cutsave_split'
test_path = os.path.join(dataset_dir, 'val')

# 载入测试集
test_dataset = datasets.ImageFolder(test_path, test_transform)
print('测试集图像数量:', len(test_dataset))
print('类别个数:', len(test_dataset.classes))
print('各类别名称:', test_dataset.classes)

# 载入类别名称 和 ID索引号 的映射字典
idx_to_labels = np.load('/media/zy/600b4f4d-dc6c-4b45-8c58-476ec9afb3fb/cyy/work/图像分类/3-【Pytorch】迁移学习训练自己的图像分类模型/idx_to_labels.npy', allow_pickle=True).item()
classes = list(idx_to_labels.values())
print('类别名称:', classes)

# 载入模型
model_path = '/media/zy/600b4f4d-dc6c-4b45-8c58-476ec9afb3fb/cyy/work/图像分类/3-【Pytorch】迁移学习训练自己的图像分类模型/checkpoints/best-0.926.pth'
model = torch.load(model_path).eval().to(device)

# 创建一个空的 DataFrame 用于保存预测结果
df_pred = pd.DataFrame()

# 定义预测函数
def predict_image(img_path, model, idx_to_labels, n=5):
    img_pil = Image.open(img_path).convert('RGB')
    input_img = test_transform(img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logits 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logits 进行 softmax 运算

    # 获取 top-n 预测结果
    top_n = torch.topk(pred_softmax, n)
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()

    pred_dict = {}
    for i in range(n):
        pred_dict[f'top-{i+1}-预测ID'] = pred_ids[i]
        pred_dict[f'top-{i+1}-预测名称'] = idx_to_labels[pred_ids[i]]

    # 检查是否有正确的 top-n 预测
    pred_dict['top-n预测正确'] = row['标注类别ID'] in pred_ids

    # 每个类别的预测置信度
    for idx, each in enumerate(classes):
        pred_dict[f'{each}-预测置信度'] = pred_softmax[0][idx].cpu().detach().numpy()

    return pred_dict

# 遍历数据集并进行预测
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    img_path = row['图像路径']
    pred_dict = predict_image(img_path, model, idx_to_labels)
    df_pred = df_pred.append(pred_dict, ignore_index=True)

# 显示预测结果
import ace_tools as tools; tools.display_dataframe_to_user(name="Prediction Results", dataframe=df_pred)
