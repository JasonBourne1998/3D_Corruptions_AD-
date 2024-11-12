import numpy as np
import os
import sys
sys.path.append(os.path.abspath("/home/yifannus2023/3D_Corruptions_AD"))
from LiDAR_corruptions import (
    density_dec_global,
    lidar_crosstalk_noise,
    cutout_local,
    fov_filter,
    fulltrajectory_noise,  # Motion Compensation
    spatial_alignment_noise,
    temporal_alignment_noise
)

class LiDARCorruption:
    def __init__(self, corruption_type, severity):
        self.corruption_type = corruption_type
        self.severity = severity
        self.corruption_map = {
            "density_decrease": density_dec_global,
            "lidar_crosstalk": lidar_crosstalk_noise,
            "cutout": cutout_local,
            "fov_lost": fov_filter,
            # "motion_compensation": fulltrajectory_noise,
            "spatial_misalignment": spatial_alignment_noise,
            "temporal_misalignment": temporal_alignment_noise
        }

    def apply_corruption(self, lidar_data, bbox_data=None, ego_pose=None):
        if self.corruption_type in self.corruption_map:
            if self.corruption_type == "motion_compensation" and ego_pose is not None:
                # Motion compensation requires ego pose
                return self.corruption_map[self.corruption_type](lidar_data, ego_pose, self.severity)
            elif self.corruption_type == "spatial_misalignment":
                # Spatial misalignment only needs severity and initial pose
                return self.corruption_map[self.corruption_type](ego_pose, self.severity)
            elif self.corruption_type == "temporal_misalignment":
                # Temporal misalignment only uses severity
                return self.corruption_map[self.corruption_type](self.severity)
            else:
                # Apply general corruption
                return self.corruption_map[self.corruption_type](lidar_data, self.severity)
        else:
            raise ValueError(f"Unknown corruption type: {self.corruption_type}")

# Load and apply corruptions to NuScenes dataset LiDAR data
def process_nuscenes_lidar_data(data_dir, output_dir, severity):
    # Check if output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over LiDAR files in the dataset
    lidar_files = [f for f in os.listdir(data_dir) if f.endswith('.pcd.bin')]
    for lidar_file in lidar_files:
        lidar_path = os.path.join(data_dir, lidar_file)
        lidar_data = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)  # Assuming 5 channels in NuScenes

        # Apply each corruption
        for corruption_name in ["density_decrease", "lidar_crosstalk", "cutout", "fov_lost", "spatial_misalignment", "temporal_misalignment"]:
            corruption = LiDARCorruption(corruption_name, severity)
            
            if corruption_name in ["motion_compensation", "spatial_misalignment"]:
                # For motion and alignment corruptions, generate a dummy ego pose
                ego_pose = np.eye(4)  # Identity matrix as placeholder for ego pose
                corrupted_data = corruption.apply_corruption(lidar_data, ego_pose=ego_pose)
            else:
                corrupted_data = corruption.apply_corruption(lidar_data)
            
            # Check if corrupted_data is a numpy array before saving
            if isinstance(corrupted_data, np.ndarray):
                output_path = os.path.join(output_dir, f"{corruption_name}_{lidar_file}")
                corrupted_data.astype(np.float32).tofile(output_path)  # Ensure saving in binary format
                print(f"Applied {corruption_name} to {lidar_file}, saved to {output_path}")
            else:
                print(f"Skipping save for {corruption_name} on {lidar_file} due to incompatible output type: {type(corrupted_data)}")

if __name__ == "__main__":
    # Define dataset path and output path
    data_dir = '/home/yifannus2023/3D_Corruptions_AD/OpenPCDet/OpenPCDet/data/nuscenes/v1.0-mini/samples/LIDAR_TOP'
    output_dir = '/home/yifannus2023/3D_Corruptions_AD/OpenPCDet/OpenPCDet/data/nuscenes_corrupted'
    severity = 3  # Adjust severity level as needed

    # Process and apply corruptions
    process_nuscenes_lidar_data(data_dir, output_dir, severity)
    print("LiDAR corruption process completed.")