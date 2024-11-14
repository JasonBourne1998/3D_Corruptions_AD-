import os
import sys
sys.path.append(os.path.abspath("/home/yifannus2023/3D_Corruptions"))
sys.path.insert(0, "/home/yifannus2023/3D_Corruptions/TransFusion/mmdet3d")
import cv2
import numpy as np

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

class ImageAddShear:
    def __init__(self, severity):
        self.shear_factor = severity * 0.1

    def __call__(self, image):
        h, w = image.shape[:2]
        shear_matrix = np.array([[1, self.shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        image_sheared = cv2.warpAffine(image, shear_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return image_sheared

class ImageAddScale:
    def __init__(self, severity):
        self.scale_factor = 1 + severity * 0.1

    def __call__(self, image):
        h, w = image.shape[:2]
        image_scaled = cv2.resize(image, (int(w * self.scale_factor), int(h * self.scale_factor)))
        return image_scaled

class ImageAddRotation:
    def __init__(self, severity):
        self.angle = severity * 15

    def __call__(self, image):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.angle, 1.0)
        image_rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return image_rotated

input_folder = '/home/yifannus2023/3D_Corruptions/TransFusion/data/kitti/testing/image_2'
output_base_folder = '/home/yifannus2023/3D_Corruptions/TransFusion/data/kitti/testing/'

num_images = 10  # 每种扰动要生成的图片数量

# 运行 severity 从 1 到 5 的扰动
for severity in range(1, 11,3):
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
        # "bbox_operation": ImageBBoxOperation(severity),
        "shear": ImageAddShear(severity),
        "scale": ImageAddScale(severity),
        "rotation": ImageAddRotation(severity)
    }

    for name, distortion in distortions.items():
        severity_folder = f"{name}_severity_{severity}"
        output_folder = os.path.join(output_base_folder, severity_folder)
        
        # 创建或检查输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 选择 input_folder 中的前10张图片
        image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))][:num_images]
        
        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)
            
            # 如果是 ImagePointAddSun
            if isinstance(distortion, ImagePointAddSun):
                lidar2img = np.eye(4)  # 示例矩阵
                distorted_image = distortion(image, lidar2img)
            else:
                distorted_image = distortion(image)
                
            # 保存扰动后的图像到对应的输出文件夹
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, distorted_image)
        
        print(f"{name} severity {severity} 处理完成并保存到 {output_folder}")

print("所有图像已经成功进行不同 severity 扰动处理并保存到相应的文件夹")