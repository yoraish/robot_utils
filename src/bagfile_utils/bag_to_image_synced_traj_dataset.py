'''
Author: Yorai Shaoul 
Date: December 2022

Create a trajectory dataset from a ROS bagfile. The saved data are images, inertial data, and motion capture data. The motion capture data that is saved is the raw est_pose poses in the est_pose frame, and also the same poses transformed to a specified base_link frame.

The dataset is saved in the following format:
    /output_root_directory
        /images
            000000000.png
            000000001.png
            ...

        /imu
            /imu.txt
                In the form:
                    timestamp, ax, ay, az, wx, wy, wz

        /est_pose
            /est_pose.txt
                In the form:
                    timestamp, x, y, z, qx, qy, qz, qw.

        /traj
            traj.txt
                In the form:
                    timestamp, x, y, z, qx, qy, qz, qw.
        
        /stamps
            stamps.txt
                In the form:
                    ix timestamp (where ix is the index of the image  and timestamp is the timestamp of the image in seconds)



Run with a command like:

python3 bag_to_image_synced_traj_dataset.py  -b /media/yoraish/overflow/data/2023-01-25_atalef_data/hover_2023-01-24-23-23-26.bag \
                                --image-topic /camera_image0 \
                                --imu-topic /imu/data \
                                --est_pose-topic /turtlebot3/motion \
                                -o /media/yoraish/overflow/data/2023-01-25_atalef_data/fwd_bwd \
                                -e 25 \
                                -s 0
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

class BagToImageSyncedTrajDataset(object):
    def __init__(self, bagfile_gp, image_topic, imu_topic, est_pose_topic, out_dir_gp = None, bagstart = 0, bagend = np.inf):
        # Parse arguments.

        self.bagfile_gp = bagfile_gp
        self.image_topic = image_topic
        self.imu_topic = imu_topic
        self.est_pose_topic = est_pose_topic

        # If no output directory specified, save as a subdirectory of the bagfile directory.
        if out_dir_gp == None:
            self.out_dir_gp = os.path.dirname(self.bagfile_gp)
        else:
            self.out_dir_gp = out_dir_gp
        self.bagstart = bagstart
        self.bagend = bagend

        # The transformations between:
        # * the est_pose motion frame (the frame who's motion is reported in the est_pose topics) and the robot frame (defined to be the IMU frame).
        # * the imu frame and the baselink frame (defined to be the same -- identity).
        # * the baselink frame and the camera frame.
        self.baselink_in_cap = np.eye(4)
        self.cam_in_baselink = np.eye(4)
        self.imu_in_baselink = np.eye(4)

        # Set the rotation and translation of the base_link frame in the cap frame.
        self.baselink_in_cap[:3,:3] = Rotation.from_euler('xyz', [0, 0, 90], degrees=True).as_matrix()

        # Create output directories.
        if not os.path.exists(self.out_dir_gp):
            os.makedirs(self.out_dir_gp)

        # Get the camera name from the image topic.
        self.cam_name = self.image_topic.split("/")[1]

        # Create output subdirectories.
        self.out_dir_images = os.path.join(self.out_dir_gp, "image_" + self.cam_name)
        if not os.path.exists(self.out_dir_images):
            os.makedirs(self.out_dir_images)
        self.out_dir_imu = os.path.join(self.out_dir_gp, "imu")
        if not os.path.exists(self.out_dir_imu):
            os.makedirs(self.out_dir_imu)
        self.out_dir_est_pose = os.path.join(self.out_dir_gp, "est_pose")
        if not os.path.exists(self.out_dir_est_pose):
            os.makedirs(self.out_dir_est_pose)
        self.out_dir_stamps = os.path.join(self.out_dir_gp, "stamps")
        if not os.path.exists(self.out_dir_stamps):
            os.makedirs(self.out_dir_stamps)
        # The trajectory text file is saved in the root directory directly.
        self.out_dir_traj = self.out_dir_gp

        # Create output files.
        self.out_file_imu = os.path.join(self.out_dir_imu, "imu.txt")
        self.out_file_est_pose = os.path.join(self.out_dir_est_pose, "est_pose.txt")
        self.out_file_traj = os.path.join(self.out_dir_traj, f"pose_{self.cam_name}.txt")
        self.out_file_stamps = os.path.join(self.out_dir_stamps, "stamps.txt")

        # Storage objects for collected information.
        self.imu_data = []     #  Of form: [stamp in nsecs, ax, ay, az, wx, wy, wz ].
        self.est_pose_data = []   #  Of form: [stamp in nsecs, x, y, z, qx, qy, qz, qw].
        self.gt_traj_data = [] #  Of form: [stamp in nsecs, x, y, z, qx, qy, qz, qw].
        self.img_stamps = []   #  Of form: [ix, stamp in nsecs].
        self.img_ix = 0
        # A pose is only added to the dataset if a frame is received. This pose is the most recent pose from the est_pose system. Under the assumption that the rate of the est_pose is high.
        self.recent_est_pose_pose = None


    def process_bag(self):
        bag = rosbag.Bag(self.bagfile_gp)
        startt = None
        print("Topics in bag: ", bag.get_type_and_topic_info()[1].keys())
        for topic, msg, t in tqdm(bag.read_messages(topics=[self.image_topic, self.imu_topic, self.est_pose_topic])):

            # Assuming that the topics are continuously published on, then the first message on any topic will mark the start of the bagfile.
            if startt is None:
                startt = t.to_sec()

            if t.to_sec() - startt > self.bagstart:
                ##################
                # Process images #
                ##################
                if topic == self.image_topic:
                    # Write the most recent est_pose pose to the dataset.
                    if self.recent_est_pose_pose is not None:
                        self.gt_traj_data.append(self.recent_est_pose_pose)

                    # Process image message.
                    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[:,:,:3] 
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    imgname = str(t.to_nsec()) + ".png"
                    imgname = str(self.img_ix).zfill(10) + ".png"

                    outputpath = os.path.join(self.out_dir_images, imgname)
                    cv2.imwrite(outputpath, img)

                    # Record the image stamp.
                    self.img_stamps.append([self.img_ix, t.to_nsec()])
                    self.img_ix += 1

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
                # Process est_pose.#
                ##################
                elif topic == self.est_pose_topic:
                    # Process est_pose message.
                    # Record the est_pose data.
                    self.est_pose_data.append([t.to_nsec(), msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

                    # Get the tracked cap pose in the est_pose-world frame.
                    cap_in_est_pose = np.eye(4)
                    cap_in_est_pose[:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
                    cap_in_est_pose[:3, :3] = Rotation.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]).as_matrix()

                    # Transform the cap pose to the robot baselink frame.
                    baselink_in_est_pose = cap_in_est_pose @ self.baselink_in_cap

                    # Convert to quaternion and translation.
                    q = Rotation.from_matrix(baselink_in_est_pose[:3,:3]).as_quat()
                    dx, dy, dz = baselink_in_est_pose[:3, 3]

                    # Record the trajectory data.
                    self.recent_est_pose_pose = [t.to_nsec(), dx, dy, dz, q[0], q[1], q[2], q[3]]

            else:
                continue
            
            # If exceeded the timeframe requested, then break.
            if t.to_sec() - startt > self.bagend:
                break

        bag.close()

        # Save the image stamps to file and include an index.
        self.img_stamps = np.array(self.img_stamps)
        np.savetxt(self.out_file_stamps, self.img_stamps, fmt="%d", header="ix nsecs")


        # Save the imu data and est_pose data to files.
        if self.imu_data:
            self.imu_data = np.vstack(self.imu_data)
            np.savetxt(self.out_file_imu, self.imu_data, fmt="%d %f %f %f %f %f %f", header="nsecs ax ay az wx wy wz")
        else:
            print(Fore.RED + "No imu data found in bagfile." + Style.RESET_ALL)
        
        if self.est_pose_data:
            self.est_pose_data = np.vstack(self.est_pose_data)
            np.savetxt(self.out_file_est_pose, self.est_pose_data, fmt="%d %f %f %f %f %f %f %f", header="nsecs x y z qx qy qz qw")

            self.gt_traj_data = np.vstack(self.gt_traj_data)
            np.savetxt(self.out_file_traj, self.gt_traj_data, fmt="%d %f %f %f %f %f %f %f", header="nsecs x y z qx qy qz qw")

    def visualize_traj(self):
        # Visualize the trajectory.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.gt_traj_data[:,1], self.gt_traj_data[:,2], self.gt_traj_data[:,3], label="Traj - base_link")
        ax.plot(self.est_pose_data[:,1], self.est_pose_data[:,2], self.est_pose_data[:,3], label="est_pose")
        
        for ix, pose in enumerate(self.gt_traj_data):
            if ix % 10 != 0:
                continue
            pose_traj = self.gt_traj_data[ix]
            u, v, w = Rotation.from_quat(pose_traj[4:]).as_matrix() @ np.array([1, 0, 0])

            pose_est_pose = self.est_pose_data[ix]
            ax.quiver(pose_traj[1], pose_traj[2], pose_traj[3], u, v, w, length=0.1, normalize=True, color="red")
            
            u, v, w = Rotation.from_quat(pose_est_pose[4:]).as_matrix() @ np.array([1, 0, 0])
            ax.quiver(pose_est_pose[1], pose_est_pose[2], pose_est_pose[3], u, v, w, length=0.1, normalize=True, color="cyan")

        ax.legend()
        plt.show()


    
def handle_args():
    # Set up command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bagfile', required=True, help='Path to the bagfile.')
    parser.add_argument('--image-topic', required=True, help='topic to extract images from.')
    parser.add_argument('--imu-topic', required=True, help='topic to extract inertial data from.')
    parser.add_argument('--est_pose-topic', required=True, help='topic to extract motion capture data from.')
    parser.add_argument('-o', '--output_root', help='Path to the output directory.')
    parser.add_argument('-s', '--start', default='13')
    parser.add_argument('-e', '--end', type=float, default= float('inf'))
    
    return parser.parse_args()

if __name__ == "__main__":
    args = handle_args()
    b2f = BagToImageSyncedTrajDataset(args.bagfile, args.image_topic, args.imu_topic, args.est_pose_topic, args.output_root, float(args.start), float(args.end))
    b2f.process_bag()
    # b2f.visualize_traj()

