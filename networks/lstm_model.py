# Author: Abhinav Gupta ag6@andrew.cmu.edu
#!/usr/bin/env python3

import sys
sys.path.append("/home/ros_ws/")
sys.path.append("/home/ros_ws/dataset")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import DiffusionDataset
from dataset.data_utils import *
from tqdm.auto import tqdm
import collections
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video

class LSTMTrainer(nn.Module):

    def __init__(self,):

        super().__init__()

        print("DUMMY LSTM TRAINER")