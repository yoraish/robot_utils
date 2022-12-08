'''
Author: Yorai Shaoul 
Date: December 2022

Save a bagfile from a sequence of image files. The names of the files will be used as the ROS-timestamp. The directory pointed at by the script should only include images.
Some code was inspired by the Kalibr implementation for a similar pipeline.

Run with a command like:

python3 images_to_bag.py -i /home/user/data/2022-11-29_project/images/ -t /turtlebot3/camera1/image_raw -o /path/to/output/mybagfile.bag
'''
from tqdm import tqdm
import rosbag
import cv2
import argparse
import numpy as np
import os
import rospy

from sensor_msgs.msg import Image

class ImagesToBag(object):
    def __init__(self):

        # Set up command line arguments.
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-i', '--images-dir', required=True)
        self.parser.add_argument('-t', '--topic', required=True)
        self.parser.add_argument('-o', '--output-bagfile-path', default='/home/outbag.bag')

        self.args = self.parser.parse_args()
        self.bagfile_gp = self.args.output_bagfile_path
        self.image_topic = self.args.topic
        self.imgs_dir_gp = self.args.images_dir

        self.counter = 0

    def write_bag(self):

        # Get the names of all the images in the directory.
        img_fns = os.listdir(self.imgs_dir_gp)

        # Convert to global paths.
        img_gps = [os.path.join(self.imgs_dir_gp, fn) for fn in img_fns]

        # The timestamps.
        img_stamps = [rospy.Time(secs = int(fn.split(".")[0][:-9]), nsecs = int(fn.split(".")[0][-9:])) for fn in img_fns] 
        # Initialize a bagfile object.
        bag = rosbag.Bag(self.bagfile_gp, 'w')

        for img_gp, img_stamp in tqdm(zip(img_gps, img_stamps)):
            # Get the image.
            img = cv2.imread(img_gp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            
            if c != 3:
                # TODO(yoraish): This is for rgba or grayscale. Deal with this later.
                raise NotImplementedError

            # Create an image message.
            msg = Image()
            msg.data = img.tobytes()
            msg.header.stamp = img_stamp
            msg.height = h
            msg.width = w
            msg.encoding = 'rgb8'
            byteperpixel = 1
            msg.step = w * c * byteperpixel 

            # Write to bag.
            bag.write(topic = self.image_topic, msg = msg, t = img_stamp)
        bag.close()

if __name__ == "__main__":
    b2f = ImagesToBag()
    b2f.write_bag()