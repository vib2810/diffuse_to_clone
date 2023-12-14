# Author: Vibhakar Mohta vib2810@gmail.com
#!/usr/bin/env python3

import pickle
import numpy as np
import torch
import sys
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as T_f
sys.path.append("/home/ros_ws/")
from src.il_packages.manipulation.src.moveit_class import get_mat_norm, get_posestamped, get_pose_norm

# Adapted from https://sites.google.com/view/playitbyear/
class AudioEncoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, audio_steps, audio_bins):
        super().__init__()
        self.audio_steps = audio_steps #how many timesteps (~57)
        self.audio_bins = audio_bins #how many bins (~100)

        self.audioConvs = nn.Sequential(
            nn.Conv1d(self.audio_bins, 64, kernel_size = 7),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size = 7),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size = 7),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size = 7),
            nn.ReLU(),
        )
        self.out_dim = 8 * (self.audio_steps - 24)

    # 1D convolution acts on the audio_bins dimension, which correlates with frequency
    def audio_forward_conv(self, audio):
        # Audio shape: (batch_size, audio_steps, audio_bins) = (batch_size, 57, 100)
        audio = audio.transpose(1, 2) # (batch_size, audio_bins, audio_steps)
        h = self.audioConvs(audio) # (batch_size, 8, audio_steps - 24)
        h = h.view(h.size(0), -1) # (batch_size, 8 * (audio_steps - 24))
        return h

    def forward(self, audio):
        #forward prop
        h_aud = self.audio_forward_conv(audio)
        return h_aud