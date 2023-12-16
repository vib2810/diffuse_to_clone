#!/usr/bin/env python3
import time
import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from audio_common_msgs.msg import AudioData
# from dataset.preprocess_audio import process_audio

class Audio:
    def __init__(self):
        self.buffer_size = 32000*3
        self.total_data = [0] * self.buffer_size

        rospy.init_node('audio_node')
        rospy.Subscriber('/audio/audio', AudioData, self.audio_callback)

        # self.im = plt.imshow(np.zeros((57, 100)), cmap='hot')
        # plt.clim(0, 1)
        # plt.colorbar()
        # plt.show(block=False)
        # plt.pause(0.001)

    def audio_callback(self, data):
        self.total_data.extend(data.data)
        if len(self.total_data) > self.buffer_size:
            self.total_data = self.total_data[-self.buffer_size:]

    def run(self):
        # while not rospy.is_shutdown():
            # print(f"Total data length: {len(self.total_data)}")
            # processed_audio = process_audio(np.array(self.total_data), check_valid=False)
            # print(f"Mean of processed audio {np.mean(processed_audio)}")
            # self.im.set_array(processed_audio)  
            # plt.show(block=False)
            # plt.pause(0.001)
            # rate.sleep()
        
        # wait for 11 seconds
        time.sleep(11)
        # save audio as npy
        np.save('/home/ros_ws/dataset/audio_classes/test.npy', np.array(self.total_data))

if __name__ == '__main__':
    audio = Audio()
    audio.run()
