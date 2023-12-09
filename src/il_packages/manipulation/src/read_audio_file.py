import os
import numpy as np

def read_npy_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                data = np.load(file_path)
                assert data.shape[0] == 32000, f"Data shape is {data.shape}"

# Example usage
folder_path = "/home/vib2810/diffuse_to_clone/logs/recorded_trajectories/vision_audio_two_blocks"
read_npy_files(folder_path)
