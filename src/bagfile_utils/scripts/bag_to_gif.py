'''
Author: Yorai Shaoul 
Date: December 2022

Save images from a ROS topic into an animated gif.

Run with a command like:

python3 bag_to_gif.py -b /home/user/data/2022-11-29_project/2022-11-28-21-24-16.bag -t /turtlebot3/camera1/image_raw -e 13.5 -s 12

Dependencies:
moviepy

'''
import rosbag
import cv2
import argparse
import numpy as np
import os
from moviepy.editor import ImageSequenceClip
import imageio

class BagToGif(object):
    def __init__(self):

        # Set up command line arguments.
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-b', '--bagfile', required=True)
        self.parser.add_argument('-t', '--topic', required=True)
        self.parser.add_argument('-o', '--output_gif_path', default='/home')
        self.parser.add_argument('-s', '--start', default='13')
        self.parser.add_argument('-e', '--end', default='15')
        self.parser.add_argument('--scale', default='1.0')
        self.parser.add_argument('--fps', default='10')
        self.args = self.parser.parse_args()
        
        self.bagfile_gp = self.args.bagfile
        self.image_topic = self.args.topic
        self.out_gif_gp = self.args.output_gif_path
        self.bagstart = float(self.args.start)
        self.bagend = float(self.args.end)
        self.scale = float(self.args.scale)
        self.fps = int(self.args.fps)

        self.imgs = []

    def process_bag(self):
        self.imgs = []
        bag = rosbag.Bag(self.bagfile_gp)
        startt = None
        for topic, msg, t in bag.read_messages(topics=[self.image_topic]):
            # Assuming that the topic is continuously published on, then the first message on the topic will mark the start of the bagfile.
            if startt is None:
                startt = t.to_sec()
            if t.to_sec() - startt > self.bagstart:
                # Process image message.
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[:,:,:3] 
                self.imgs.append(img)

            else:
                continue
            
            if t.to_sec() - startt > self.bagend:
                break
        bag.close()

    def imgs_to_gif(self):
        """Code inspired by https://stackoverflow.com/questions/13041893/generating-an-animated-gif-in-python.
        """
        clip = ImageSequenceClip(self.imgs, fps=self.fps).resize(self.scale)
        clip.write_gif(self.out_gif_gp, fps=self.fps)
        return clip

if __name__ == "__main__":
    b2g = BagToGif()
    b2g.process_bag()
    b2g.imgs_to_gif()