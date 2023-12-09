#!/usr/bin/env python3
import time
import rospy
from audio_common_msgs.msg import AudioData

class Audio:
    def __init__(self):
        self.total_data = []
        self.buffer_size = 30000
        rospy.init_node('audio_node')
        rospy.Subscriber('/audio/audio', AudioData, self.audio_callback)

    def audio_callback(self, data):
        self.total_data.extend(data.data)
        if len(self.total_data) > self.buffer_size:
            self.total_data = self.total_data[-self.buffer_size:]

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            print(f"Total data length: {len(self.total_data)}")
            rate.sleep()

if __name__ == '__main__':
    audio = Audio()
    audio.run()
