from nnunetv2.training.nnUNetTrainer.STUNetTrainer import STUNetTrainer_small as nnUNetTrainer
from torchinfo import summary
import json
from fvcore.nn import FlopCountAnalysis
import torch


# 读取并解析 dataset_json 文件
with open("/scratch/users/k23065445/baseline/nnUNet_preprocessed/Dataset277_TotalSegmentator/dataset.json", 'r') as f:
    dataset_json_dict = json.load(f)

# 读取并解析 nnUNetPlans.json 文件
with open("/scratch/users/k23065445/baseline/nnUNet_preprocessed/Dataset277_TotalSegmentator/nnUNetPlans.json", 'r') as f:
    nnUNetPlans_dict = json.load(f)

# 创建并初始化模型
model = nnUNetTrainer(
    plans=nnUNetPlans_dict,
    configuration="3d_fullres",
    fold=277,
    dataset_json=dataset_json_dict,
)
model.initialize()

# 使用随机输入数据来计算参数量和 FLOPs
input_size = (1, 1, 112, 112, 128)  # 根据你的输入数据尺寸进行调整
summary(model.network, input_size=input_size)

# 计算 FLOPs
random_input = torch.randn(input_size).cuda()
flops = FlopCountAnalysis(model.network, random_input)
print(f"FLOPs: {flops.total()/1e9:.1f} G")
