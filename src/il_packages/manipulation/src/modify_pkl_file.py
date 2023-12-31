import os
import shutil

folder_path = "/home/ros_ws/logs/recorded_trajectories/vision_audio_coins"
output_folder_path = "/home/ros_ws/dataset/data/vision_audio_coins"
convert_audio = True

total_num_of_files = len(sorted(os.listdir(folder_path)))
split_ratio = 0.8

# make the output_folder_path
os.makedirs(output_folder_path, exist_ok=True)

# make train and eval folders:
os.makedirs(os.path.join(output_folder_path, "train"), exist_ok=True)
os.makedirs(os.path.join(output_folder_path, "eval"), exist_ok=True)

# make the train and eval and images folder for output_folder_path
os.makedirs(os.path.join(output_folder_path, "train/Images"), exist_ok=True)
os.makedirs(os.path.join(output_folder_path, "eval/Images"), exist_ok=True)

# make the train and eval and Audio folder for output_folder_path
if convert_audio:
    os.makedirs(os.path.join(output_folder_path, "train/Audio"), exist_ok=True)
    os.makedirs(os.path.join(output_folder_path, "eval/Audio"), exist_ok=True)

for file_name in sorted(os.listdir(folder_path)):
  if file_name.endswith(".pkl"):
      
    trajectory_id = file_name.split("_")[-1].split(".")[0]
    print(file_name.split("_")[-1].split(".")[0])

    pickle_file_path = os.path.join(folder_path, file_name)
    images_file_path = os.path.join(folder_path, "Images/" + trajectory_id)
    if convert_audio:
        audio_file_path = os.path.join(folder_path, "Audio/" + trajectory_id)

    if int(trajectory_id) < total_num_of_files * split_ratio:
        print("Trajectory ID: ", trajectory_id)
        print("In train")
        output_trajectory_id = len(os.listdir(os.path.join(output_folder_path, "train/Images")))
        
        output_pickle_path = os.path.join(output_folder_path, "train/" + str(output_trajectory_id) + ".pkl")
        output_images_path = os.path.join(output_folder_path, "train/Images/" + str(output_trajectory_id))
        if convert_audio:
            output_audio_path = os.path.join(output_folder_path, "train/Audio/" + str(output_trajectory_id))
    else:
        print("Trajectory ID: ", trajectory_id)
        print("In eval")
        output_trajectory_id = len(os.listdir(os.path.join(output_folder_path, "eval/Images")))
        
        output_pickle_path = os.path.join(output_folder_path, "eval/" + str(output_trajectory_id) + ".pkl")
        output_images_path = os.path.join(output_folder_path, "eval/Images/" + str(output_trajectory_id))
        if convert_audio:
            output_audio_path = os.path.join(output_folder_path, "eval/Audio/" + str(output_trajectory_id))

    shutil.copy(pickle_file_path, output_pickle_path)
    shutil.copytree(images_file_path, output_images_path)
    if convert_audio:
        shutil.copytree(audio_file_path, output_audio_path)
    
