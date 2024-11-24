import albumentations as A
from albumentations.augmentations.transforms import RandomRain, RandomFog
from albumentations.core.composition import Compose
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

import os
import cv2
import albumentations as A
import matplotlib.pyplot as plt
from tqdm import tqdm

# 定义增强管道
night_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.35, -0.1), contrast_limit=(-0.3, 0), p=1),  # 调整亮度和对比度
    A.RGBShift(r_shift_limit=(-20, 0), g_shift_limit=(-20, 0), b_shift_limit=(10, 40), p=1),  # 冷色调
    A.MotionBlur(blur_limit=(3, 5), p=0.5)  # 模糊效果
])

rain_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0), contrast_limit=(-0.3, 0.2), p=1),  # 降低亮度和对比度
    A.RandomRain(
        slant_lower=-10, slant_upper=10,  # 增大雨滴的倾斜角度范围
        drop_length=25,  # 增加雨滴长度
        drop_width=2,  # 增加雨滴宽度，确保更明显
        drop_color=(180, 180, 180),  # 雨滴颜色稍暗，接近自然雨水颜色
        blur_value=2,  # 模糊程度降低以增加清晰度
        brightness_coefficient=0.7,  # 整体画面亮度降低
        p=1
    ),
    A.MotionBlur(blur_limit=(3, 5), p=0.3),  # 可选：增加轻微模糊，模拟降雨时的动态效果
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # 可选：模拟环境模糊
])

fog_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.1), contrast_limit=(-0.3, -0.1), p=1),  # 模拟光线减弱
    A.RandomFog(
        fog_coef_lower=0.2,  # 雾浓度下界（0 表示无雾，1 表示完全被雾覆盖）
        fog_coef_upper=0.7,  # 雾浓度上界
        alpha_coef=0.3,  # 雾层透明度，较低值更贴近自然
        p=1
    ),
    A.GaussianBlur(blur_limit=(5, 7), p=0.5),  # 轻微模糊，模拟雾气对光线的散射
])

# 输入和输出文件夹
input_folder = 'D:/navigationData/test/'  # 输入文件夹，存放原始图像
output_folder = 'D:/navigationData/test_result/'  # 输出文件夹，用于保存增强后的图像
os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在

# 获取所有图片文件
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 批量处理
for image_name in tqdm(image_files, desc="Processing Images"):
    # 加载图像
    input_image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Could not read image: {image_name}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 应用增强
    augmented = night_transform(image=image)
    augmented_image = augmented['image']

    # 保存增强图像
    output_image_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_augmented.jpg")
    cv2.imwrite(output_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))  # 转换回 BGR 格式保存

    #（可选）显示原图和增强图像（仅调试用）
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image")
    # plt.imshow(image)
    # plt.axis("off")
    #
    # plt.subplot(1, 2, 2)
    # plt.title("Augmented Image")
    # plt.imshow(augmented_image)
    # plt.axis("off")
    # plt.show()

print(f"All augmented images saved to: {output_folder}")


