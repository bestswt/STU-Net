from nnunet.inference.predict import load_model_and_checkpoint_files
from torchinfo import summary
import torch

# 指定模型文件路径
model_file_path = ("/scratch/users/k23065445/baseline/nnUNet_results/Dataset277_TotalSegmentator"
                   "/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth")

# 加载模型和检查点文件
trainer, params = load_model_and_checkpoint_files(model_file_path)

# 获取模型
model = trainer.network

# 方法1：使用torchsummary
model_summary = summary(model, input_size=(2, 112, 112, 128))  # 这里的input_size应根据您的实际输入大小进行调整
print(f"Total FLOPs: {model_summary.total_ops}")


# 方法2：直接计算参数数量
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")
