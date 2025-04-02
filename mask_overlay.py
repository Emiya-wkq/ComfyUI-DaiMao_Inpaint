import torch
import numpy as np
from PIL import Image

class MaskOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "structure_preservation": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "display_name": "重绘区域结构保留强度"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "custom_nodes"
    DISPLAY_NAME = "遮罩叠加处理"

    def process(self, image, mask, structure_preservation=0.0):
        # 确保图像和遮罩在同一设备上
        device = image.device
        
        # 将遮罩调整到与图像相同的尺寸
        if mask.shape[-2:] != image.shape[-2:]:
            mask = torch.nn.functional.interpolate(
                mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
                size=(image.shape[1], image.shape[2]), 
                mode='bilinear'
            ).squeeze(1)
        
        # 二值化图像 - 使用固定阈值0.5
        # 将图像转换为灰度
        if image.shape[3] == 3:  # RGB图像
            # 使用RGB到灰度的标准转换公式
            gray_image = 0.299 * image[:, :, :, 0] + 0.587 * image[:, :, :, 1] + 0.114 * image[:, :, :, 2]
        else:  # 已经是灰度图像
            gray_image = image[:, :, :, 0]
        
        # 二值化灰度图像 - 使用固定阈值0.5
        binary_image = (gray_image > 0.5).float()
        
        # 扩展为3通道
        binary_image = binary_image.unsqueeze(-1).repeat(1, 1, 1, 3)
        
        # 创建黑色透明图像
        black_image = torch.zeros_like(image)
        
        # 创建透明度通道 (alpha)，遮罩区域为不透明，非遮罩区域为透明
        alpha = mask.unsqueeze(-1)  # 添加通道维度
        
        # 将alpha通道与黑色图像合并
        intermediate_result = black_image * alpha + binary_image * (1 - alpha)
        
        # 计算不透明度 = 1 - 结构保留强度
        opacity = 1.0 - structure_preservation
        
        # 修改透明度应用方式
        # 只在mask区域应用计算出的不透明度
        mask_opacity = mask.unsqueeze(-1) * opacity
        final_result = intermediate_result * mask_opacity + image * (1 - mask_opacity)
        
        return (final_result,) 