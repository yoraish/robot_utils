'''
Author: Yorai Shaoul 
Date: December 2022

Create a trajectory dataset from a ROS bagfile. The saved data are images, inertial data, and motion capture data. The motion capture data that is saved is the raw mocap poses in the mocap frame, and also the same poses transformed to a specified base_link frame.

The dataset is saved in the following format:
    /output_root_directory
        /images
            /000000000.png
            /000000001.png
            ...

        /imu
            /imu.txt
                In the form:
                    timestamp, ax, ay, az, wx, wy, wz

        /mocap
            /mocap.txt
                In the form:
                    timestamp, x, y, z, qx, qy, qz, qw.

        /traj
            /traj.txt
                In the form:
                    timestamp, x, y, z, qx, qy, qz, qw.



Run with a command like:

python3 bag_to_traj_dataset.py  -b /home/user/data/2022-11-29_project/2022-11-28-21-24-16.bag \
                                --image-topic /turtlebot3/camera1/image_raw \
                                --imu-topic /turtlebot3/imu/data \
                                --mocap-topic /turtlebot3/motion \
                                -o /home/user/data/2022-11-29_project/ \
                                -e 13.5 \
                                -s 12
'''

# General imports.
from colorama import Fore, Style
import cv2
import argparse
import numpy as np
import os
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from matplotlib import pyplot as plt
import pytransform3d as pt3d
from pytransform3d.rotations import *

# ROS imports.
import rosbag

class BagToTrajDataset(object):
    def __init__(self):

        # Set up command line arguments.
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-b', '--bagfile', required=True, help='Path to the bagfile.')
        self.parser.add_argument('--image-topic', required=True, help='topic to extract images from.')
        self.parser.add_argument('--imu-topic', required=True, help='topic to extract inertial data from.')
        self.parser.add_argument('--mocap-topic', required=True, help='topic to extract motion capture data from.')
        self.parser.add_argument('-o', '--output_root', default='/home', help='Path to the output directory.')
        self.parser.add_argument('-s', '--start', default='13')
        self.parser.add_argument('-e', '--end', default='0')
        
        self.args = self.parser.parse_args()

        self.bagfile_gp = self.args.bagfile
        self.image_topic = self.args.image_topic
        self.imu_topic = self.args.imu_topic
        self.mocap_topic = self.args.mocap_topic
        self.out_dir_gp = self.args.output_root
        self.bagstart = float(self.args.start)
        self.bagend = float(self.args.end)
        if not self.bagend:
            self.bagend = float('inf')


        # The transformations between:
        # * the mocap motion frame (the frame who's motion is reported in the mocap topics) and the robot frame (defined to be the IMU frame).
        # * the imu frame and the baselink frame (defined to be the same -- identity).
        # * the baselink frame and the camera frame.
        self.baselink_in_cap = np.eye(4)
        self.cam0_in_baselink = np.eye(4)
        self.imu_in_baselink = np.eye(4)

        # Set the rotation and translation of the base_link frame in the cap frame.
        self.baselink_in_cap[:3,:3] = Rotation.from_euler('xyz', [0, 0, 90], degrees=True).as_matrix()

        # Create output directories.
        if not os.path.exists(self.out_dir_gp):
            os.makedirs(self.out_dir_gp)

        # Create output subdirectories.
        self.out_dir_images = os.path.join(self.out_dir_gp, "images")
        if not os.path.exists(self.out_dir_images):
            os.makedirs(self.out_dir_images)
        self.out_dir_imu = os.path.join(self.out_dir_gp, "imu")
        if not os.path.exists(self.out_dir_imu):
            os.makedirs(self.out_dir_imu)
        self.out_dir_mocap = os.path.join(self.out_dir_gp, "mocap")
        if not os.path.exists(self.out_dir_mocap):
            os.makedirs(self.out_dir_mocap)
        self.out_dir_traj = os.path.join(self.out_dir_gp, "traj")
        if not os.path.exists(self.out_dir_traj):
            os.makedirs(self.out_dir_traj)

        # Create output files.
        self.out_file_imu = os.path.join(self.out_dir_imu, "imu.txt")
        self.out_file_mocap = os.path.join(self.out_dir_mocap, "mocap.txt")
        self.out_file_traj = os.path.join(self.out_dir_traj, "traj.txt")

        # Storage objects for collected information.
        self.imu_data = [] #   Of form: [stamp in nsecs, ax, ay, az, wx, wy, wz ].
        self.mocap_data = [] # Of form: [stamp in nsecs, x, y, z, qx, qy, qz, qw].
        self.traj_data = [] #  Of form: [stamp in nsecs, x, y, z, qx, qy, qz, qw].


    def process_bag(self):
        bag = rosbag.Bag(self.bagfile_gp)
        startt = None
        for topic, msg, t in tqdm(bag.read_messages(topics=[self.image_topic, self.imu_topic, self.mocap_topic])):

            # Assuming that the topics are continuously published on, then the first message on any topic will mark the start of the bagfile.
            if startt is None:
                startt = t.to_sec()

            if t.to_sec() - startt > self.bagstart:
                ##################
                # Process images #
                ##################
                if topic == self.image_topic:
                    # Process image message.
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[:,:,:3] 
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    imgname = str(t.to_nsec()) + ".png"
                    outputpath = os.path.join(self.out_dir_images, imgname)
                    cv2.imwrite(outputpath, img)

                ##################
                # Process imu.   #
                ##################
                elif topic == self.imu_topic:
                    # Process imu message.
                    self.imu_data.append([t.to_nsec(), 
                    msg.linear_acceleration.x, 
                    msg.linear_acceleration.y, 
                    msg.linear_acceleration.z, 
                    msg.angular_velocity.x, 
                    msg.angular_velocity.y, 
                    msg.angular_velocity.z])

                ##################
                # Process mocap.#
                ##################
                elif topic == self.mocap_topic:
                    # Process mocap message.
                    # Record the mocap data.
                    self.mocap_data.append([t.to_nsec(), msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

                    # Get the tracked cap pose in the mocap-world frame.
                    cap_in_mocap = np.eye(4)
                    cap_in_mocap[:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
                    cap_in_mocap[:3, :3] = Rotation.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]).as_matrix()

                    # Transform the cap pose to the robot baselink frame.
                    baselink_in_mocap = cap_in_mocap @ self.baselink_in_cap

                    # Convert to quaternion and translation.
                    q = Rotation.from_matrix(baselink_in_mocap[:3,:3]).as_quat()
                    dx, dy, dz = baselink_in_mocap[:3, 3]

                    # Record the trajectory data.
                    self.traj_data.append([t.to_nsec(), dx, dy, dz, q[0], q[1], q[2], q[3]]) 

            else:
                continue
            
            # If exceeded the timeframe requested, then break.
            if t.to_sec() - startt > self.bagend:
                break

        bag.close()

        # Save the imu data and mocap data to files.
        if self.imu_data:
            self.imu_data = np.vstack(self.imu_data)
            np.savetxt(self.out_file_imu, self.imu_data, fmt="%d %f %f %f %f %f %f", header="nsecs ax ay az wx wy wz")
        else:
            print(Fore.RED + "No imu data found in bagfile." + Style.RESET_ALL)
        self.mocap_data = np.vstack(self.mocap_data)
        np.savetxt(self.out_file_mocap, self.mocap_data, fmt="%d %f %f %f %f %f %f %f", header="nsecs x y z qx qy qz qw")

        self.traj_data = np.vstack(self.traj_data)
        np.savetxt(self.out_file_traj, self.traj_data, fmt="%d %f %f %f %f %f %f %f", header="nsecs x y z qx qy qz qw")

    def visualize_traj(self):
        # Visualize the trajectory.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.traj_data[:,1], self.traj_data[:,2], self.traj_data[:,3], label="Traj - base_link")
        ax.plot(self.mocap_data[:,1], self.mocap_data[:,2], self.mocap_data[:,3], label="Mocap")
        
        for ix, pose in enumerate(self.traj_data):
            if ix % 10 != 0:
                continue
            pose_traj = self.traj_data[ix]
            u, v, w = Rotation.from_quat(pose_traj[4:]).as_matrix() @ np.array([1, 0, 0])

            pose_mocap = self.mocap_data[ix]
            ax.quiver(pose_traj[1], pose_traj[2], pose_traj[3], u, v, w, length=0.1, normalize=True, color="red")
            
            u, v, w = Rotation.from_quat(pose_mocap[4:]).as_matrix() @ np.array([1, 0, 0])
            ax.quiver(pose_mocap[1], pose_mocap[2], pose_mocap[3], u, v, w, length=0.1, normalize=True, color="cyan")

        ax.legend()
        plt.show()


    

if __name__ == "__main__":
    b2f = BagToTrajDataset()
    b2f.process_bag()
    b2f.visualize_traj()

