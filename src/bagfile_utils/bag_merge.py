'''
Author: Yorai Shaoul 
Date: December 2022

Merge mutliple bagfiles into one.
python3 bag_merge.py -b /path/to/bag/one /path/to/bag/two /path/to/bag/three -s 1668625186 -e 16686251200 -o /path/to/output/mybagfile.bag

'''

from colorama import Fore
from tqdm import tqdm
import rosbag
import cv2
import argparse
import numpy as np
import os
import rospy


class BagMerger(object):
    def __init__(self):

        # Set up command line arguments.
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-b','--bagfiles', nargs='+', help='<Required> Set flag', required=True)
        self.parser.add_argument('-s', '--start', type=int, default=0)
        self.parser.add_argument('-e', '--end', type=int, default=float('inf'))
        self.parser.add_argument('-o', '--output-bagfile-path', default='')

        self.args = self.parser.parse_args()
        self.bagfiles = self.args.bagfiles
        self.start = self.args.start
        self.end = self.args.end
        self.counter = 0

        self.output_bagfile_gp = self.args.output_bagfile_path
        if self.output_bagfile_gp == '':
            self.output_bagfile_gp =  os.path.join(os.path.dirname(self.bagfiles[0]), "_".join([b.split(".")[0].split("/")[-1] for b in self.bagfiles]) + "_merged.bag")

            # Check that the file does not exist, otherwise, add a number to the end.
            if os.path.exists(self.output_bagfile_gp):
                i = 0
                while os.path.exists(self.output_bagfile_gp):
                    self.output_bagfile_gp = os.path.join(os.path.dirname(self.bagfiles[0]), "_".join([b.split(".")[0].split("/")[-1] for b in self.bagfiles]) + "_merged_{}.bag".format(i))
                    i += 1

            print(Fore.GREEN + "Output bagfile path not specified. Saving to: {}".format(self.output_bagfile_gp) + Fore.RESET)

    def write_bag(self):
        
        bag = rosbag.Bag(self.output_bagfile_gp, 'w')
        for bagfile in self.bagfiles:
            print("Processing bagfile: {}".format(bagfile))
            with rosbag.Bag(bagfile, 'r') as inbag:
                for topic, msg, t in inbag.read_messages():
                    if t.to_sec() >= self.start and t.to_sec() < self.end:
                        bag.write(topic, msg, t)
                    self.counter += 1

        bag.close()

if __name__ == "__main__":
    b2f = BagMerger()
    b2f.write_bag()