'''
Author: Yorai Shaoul 
Date: November 2022

Save images from a ROS topic in files.

Run with a command like:

python3 bag_to_images.py -b /home/user/data/2022-11-29_project/2022-11-28-21-24-16.bag -t /turtlebot3/camera1/image_raw -e 13.5 -s 12
'''
import rosbag
import cv2
import argparse
import numpy as np
import os

class BagToFiles(object):
    def __init__(self):

        # Set up command line arguments.
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-b', '--bagfile', required=True)
        self.parser.add_argument('-t', '--topic', required=True)
        self.parser.add_argument('-o', '--output_image_path', default='/home')
        self.parser.add_argument('-s', '--start', default='13')
        self.parser.add_argument('-e', '--end', default='15')
        self.args = self.parser.parse_args()
        self.bagfile_gp = self.args.bagfile
        self.image_topic = self.args.topic
        self.out_dir_gp = self.args.output_image_path
        self.bagstart = float(self.args.start)
        self.bagend = float(self.args.end)

        self.counter = 0

    def process_bag(self):
        bag = rosbag.Bag(self.bagfile_gp)
        startt = None
        for topic, msg, t in bag.read_messages(topics=[self.image_topic]):
            # Assuming that the topic is continuously published on, then the first message on the topic will mark the start of the bagfile.
            if startt is None:
                startt = t.to_sec()
            if t.to_sec() - startt > self.bagstart:
                # Process image message.
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[:,:,:3] 
                # img = np.roll(img, [0,0,1])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # img = np.transpose(img, (2,1,0))
                outputpath = os.path.join(self.out_dir_gp, str(self.counter) + ".png")
                self.counter += 1
                cv2.imwrite(outputpath, img)


            else:
                continue
            
            if t.to_sec() - startt > self.bagend:
                return
        bag.close()

if __name__ == "__main__":
    b2f = BagToFiles()
    b2f.process_bag()