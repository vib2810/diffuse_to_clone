import os
import shutil

folder_path = "/home/vib2810/diffuse_to_clone/logs/recorded_trajectories/second_vision_collect"
output_folder_path = "/home/vib2810/diffuse_to_clone/dataset/data/data_block_pick"

total_num_of_files = len(os.listdir(folder_path))
split_ratio = 0.8

for file_name in os.listdir(folder_path):
  if file_name.endswith(".pkl"):
      
    trajectory_id = file_name.split("_")[-1].split(".")[0]
    print(file_name.split("_")[-1].split(".")[0])

    pickle_file_path = os.path.join(folder_path, file_name)
    images_file_path = os.path.join(folder_path, "Images/" + trajectory_id)

    if int(trajectory_id) < total_num_of_files * split_ratio:
        output_trajectory_id = len(os.listdir(os.path.join(output_folder_path, "train/Images")))
        
        output_pickle_path = os.path.join(output_folder_path, "train/" + str(output_trajectory_id) + ".pkl")
        output_images_path = os.path.join(output_folder_path, "train/Images/" + str(output_trajectory_id))
    else:
        output_trajectory_id = len(os.listdir(os.path.join(output_folder_path, "eval/Images")))
        
        output_pickle_path = os.path.join(output_folder_path, "eval/" + str(output_trajectory_id) + ".pkl")
        output_images_path = os.path.join(output_folder_path, "eval/Images/" + str(output_trajectory_id))
    
    if not os.path.exists(output_images_path):
        os.makedirs(output_images_path)

    shutil.copy(pickle_file_path, output_pickle_path)
    shutil.copytree(images_file_path, output_images_path)
    
