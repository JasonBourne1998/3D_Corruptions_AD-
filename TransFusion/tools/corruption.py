import os
import sys
sys.path.append(os.path.abspath("/home/yifannus2023/3D_Corruptions"))
sys.path.insert(0, "/home/yifannus2023/3D_Corruptions/TransFusion/mmdet3d")
# sys.path.append('/home/yifannus2023/3D_Corruptions_AD-/utils')
import cv2
import numpy as np
from torchvision.utils import save_image

# 导入您的现有扰动类和新的扰动类
from Camera_corruptions import (
    ImageAddFog,
    ImageAddSnow,
    ImageAddRain,
    ImageAddGaussianNoise,
    ImageAddImpulseNoise,
    ImageAddUniformNoise,
    ImageMotionBlurFrontBack,
    ImageMotionBlurLeftRight,
    ImagePointAddSun,
    ImageBBoxOperation
)

# 定义新的扰动类
class ImageAddShear:
    def __init__(self, severity):
        self.shear_factor = severity * 0.1  # 调整剪切程度

    def __call__(self, image):
        h, w = image.shape[:2]
        shear_matrix = np.array([[1, self.shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        image_sheared = cv2.warpAffine(image, shear_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return image_sheared

class ImageAddScale:
    def __init__(self, severity):
        self.scale_factor = 1 + severity * 0.1  # 调整缩放比例

    def __call__(self, image):
        h, w = image.shape[:2]
        image_scaled = cv2.resize(image, (int(w * self.scale_factor), int(h * self.scale_factor)))
        return image_scaled

class ImageAddRotation:
    def __init__(self, severity):
        self.angle = severity * 15  # 每级别增加15度

    def __call__(self, image):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        image_rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return image_rotated

# 定义输入和输出文件夹路径
input_folder = '/home/yifannus2023/3D_Corruptions_AD/TransFusion/data/kitti/testing/image_2'
output_base_folder = '/home/yifannus2023/3D_Corruptions_AD/TransFusion/data/kitti/testing/'

# 设置扰动的严重程度
severity = 3  # 可以根据需要调整这个值

# 创建扰动效果的实例
distortions = {
    "fog": ImageAddFog(severity, seed=42),
    "snow": ImageAddSnow(severity, seed=42),
    "rain": ImageAddRain(severity, seed=42),
    "sun": ImagePointAddSun(severity),
    "gaussian_noise": ImageAddGaussianNoise(severity, seed=42),
    "impulse_noise": ImageAddImpulseNoise(severity, seed=42),
    "uniform_noise": ImageAddUniformNoise(severity),
    "motion_blur_fb": ImageMotionBlurFrontBack(severity),
    "motion_blur_lr": ImageMotionBlurLeftRight(severity),
    "bbox_operation": ImageBBoxOperation(severity),
    "shear": ImageAddShear(severity),  # 新增剪切
    "scale": ImageAddScale(severity),  # 新增缩放
    "rotation": ImageAddRotation(severity)  # 新增旋转
}

# 对每种扰动生成10张图片
num_images = 10  # 每种扰动要生成的图片数量

for name, distortion in distortions.items():
    output_folder = os.path.join(output_base_folder, name)
    
    # 检查该扰动的输出文件夹是否已经存在
    if os.path.exists(output_folder):
        print(f"{name} 已经存在，跳过...")
        continue
    else:
        os.makedirs(output_folder, exist_ok=True)
        
        # 遍历 input_folder 中的所有图像文件，生成10张图片
        image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))][:num_images]
        
        for image_file in image_files:
            # 读取图像
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)
            
            # 如果是 ImagePointAddSun，只需要 image 和 lidar2img 矩阵
            if isinstance(distortion, ImagePointAddSun):
                lidar2img = np.eye(4)  # 示例矩阵
                distorted_image = distortion(image, lidar2img)
            else:
                # 其他扰动只处理图像
                distorted_image = distortion(image)
                
            # 保存扰动后的图像到对应的输出文件夹
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, distorted_image)
    
    print(f"{name} 处理完成并保存到 {output_folder}")

print("所有图像已经成功进行扰动处理并保存到相应的文件夹")