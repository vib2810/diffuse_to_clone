import pickle

pickle_file_path = '/home/ros_ws/dataset/data/data/0.pkl'

with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

print(data)