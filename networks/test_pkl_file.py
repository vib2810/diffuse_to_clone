# read all files that end with .pkl in the directory
# /home/ros_ws/logs/recorded_trajectories/<expt_name>/
import os
import pickle
import numpy as np

# expt_name = "toy_expt_first_try"
expt_name = "toy_expt_vision_block_pick"

root = "/home/ros_ws/logs/recorded_trajectories/"
file_names= []
# iterate through all files in the directory
for file in os.listdir("/home/ros_ws/logs/recorded_trajectories/" + expt_name + "/"):
    if file.endswith(".pkl"):
        file_names.append(file)
        

# print content of files in the file_names list
for file_name in file_names:
    data = pickle.load(open(root + expt_name + "/" + file_name, 'rb'))
    print("File name:", file_name)
    # print the keys in data
    for key, value in data.items():
        print("Key:", key)
        if type(value)==list:
            print("Length of list:", len(value))
        if type(value)==np.ndarray:
            print("Shape of ndarray:", value.shape)
    print("---------------------------------")
