from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torchinfo import summary

# 创建并初始化模型
model = nnUNetTrainer(
    plans="/scratch/users/k23065445/baseline/nnUNet_preprocessed/Dataset277_TotalSegmentator/nnUNetPlans.json",
    configuration="3d_fullres",
    fold=277,
    dataset_json="/scratch/users/k23065445/baseline/nnUNet_preprocessed/Dataset277_TotalSegmentator/dataset.json",
)
model.initialize()

# 使用随机输入数据来计算参数量和 FLOPs
input_size = (2, 1, 128, 128, 128)  # 根据你的输入数据尺寸进行调整
summary(model.network, input_size=input_size)
