'''
Author: Yorai Shaoul 
Date: May 2023

Create a trajectory dataset, in the tartanair format, from a ROS bagfile. The saved data are images, and motion capture data (global position in NED world frame).

The dataset is saved in the following format:
    /output_root_directory
        /MocapEnv
            /Data_hard
                /image_<new_cam_name>
                    /000000.png
                    /000001.png
                    ...
                pose_<new_cam_name>.txt

                

Run with a command like:
python3  bag_to_tartanair.py \
     -b /media/yoraish/overflow/data/fish_mocap/2023-05-09-11-42-44.bag \
     -o /media/yoraish/overflow/data/fish_mocap/tartanair_converted \
     --image-topic /camera_image1 \
     --mocap-topic /mocap_node/fish/pose \
     --hz 10 \
     -s 10 \
     --e 1200
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

class BagToTartanAir(object):
    def __init__(self, bagfile, image_topic, mocap_topic, tartanair_data_root, new_env_name, new_cam_name, hz, start = 0, end = np.inf):
        # Parse arguments.

        self.bagfile_gp = bagfile
        self.image_topic = image_topic
        self.mocap_topic = mocap_topic
        self.hz = hz

        # Check which trajectories exist, and create a new one if needed.
        existing_traj_dirs = os.listdir(os.path.join(tartanair_data_root, new_env_name, "Data_hard"))
        existing_traj_dirs = sorted([d for d in existing_traj_dirs if d.startswith("P")])

        if len(existing_traj_dirs) == 0:
            traj_name = "P000"
        else:
            traj_name = existing_traj_dirs[-1]

        # If no output directory specified, save as a subdirectory of the bagfile directory.
        self.out_dir_gp = os.path.join(tartanair_data_root, new_env_name, "Data_hard", traj_name)

        # Create if the output directory does not exist.
        if not os.path.exists(self.out_dir_gp):
            os.makedirs(self.out_dir_gp)
        else:
            # overwrite = input(Fore.RED + f"Warning: Output directory [{self.out_dir_gp}] already exists. Would you like to overwrite? [Y/n]: " + Style.RESET_ALL)
            overwrite = 'N'
            if overwrite == "Y" or overwrite == "":
                print("Overwriting.")
            else:
                traj_name = int(traj_name[1:]) + 1
                traj_name = "P" + str(traj_name).zfill(3)
                # new_traj = input(Fore.RED + f"Would you like to create a new trajectory named [{traj_name}]? [Y/n]: " + Style.RESET_ALL)
                new_traj = 'Y'
                if new_traj == "Y" or new_traj == "":
                    self.out_dir_gp = os.path.join(tartanair_data_root, new_env_name, "Data_hard", traj_name)
                    os.makedirs(self.out_dir_gp)
                else:
                    print("Exiting.")
                    exit()


        self.bagstart = float(start)
        self.bagend =   float(end)

        # The transformations between:
        # * the mocap motion frame (the frame who's motion is reported in the mocap topics) and the robot frame (defined to be the IMU frame).
        # * the imu frame and the baselink frame (defined to be the same -- identity).
        # * the baselink frame and the camera frame.
        self.baselink_in_cap = np.eye(4)
        self.cam0_in_baselink = np.eye(4)
        self.imu_in_baselink = np.eye(4)

        # Set the rotation and translation of the base_link frame in the cap frame.
        # self.baselink_in_cap[:3,:3] = Rotation.from_euler('xyz', [0, 0, 90], degrees=True).as_matrix()


        # Create output subdirectories.
        self.out_dir_images = os.path.join(self.out_dir_gp, "image_" + new_cam_name)
        if not os.path.exists(self.out_dir_images):
            os.makedirs(self.out_dir_images)

        # Create output files.
        self.out_file_pose = os.path.join(self.out_dir_gp, f"pose_{new_cam_name}.txt")
        self.out_file_pose_lcam_front = os.path.join(self.out_dir_gp, "pose_lcam_front.txt") # This is a hack to get TVOFE to be happy.
        self.out_file_image_stamps = os.path.join(self.out_dir_gp, "image_stamps.txt")

        # Storage objects for collected information.
        self.gt_pose_data = [] #  List of lists of form: [stamp in nsecs, x, y, z, qx, qy, qz, qw].
        self.image_stamps = [] # Of form: stamp-in-nsecs image_ros/000000.png.

        # Counter objects.
        self.image_counter = 0

        # Allow to register image flag. Only true when enough time has passed since the last image was registered.
        self.allow_register_image = True
        self.last_image_stamp = 0
        self.last_gt_pose = [] #  Of form: [stamp in nsecs, x, y, z, qx, qy, qz, qw].

    def process_bag(self):
        bag = rosbag.Bag(self.bagfile_gp)
        startt = None
        for topic, msg, t in tqdm(bag.read_messages(topics=[self.image_topic, self.mocap_topic])):

            # Assuming that the topics are continuously published on, then the first message on any topic will mark the start of the bagfile.
            if startt is None:
                startt = t.to_sec()


            if t.to_sec() - startt > self.bagstart:
                # Check if registering an image is allowed.
                if not self.allow_register_image:
                    if t.to_nsec() - self.last_image_stamp > 1e9 / float(self. hz):
                        self.allow_register_image = True

                ##################
                # Process images #
                ##################
                if topic == self.image_topic and self.allow_register_image:
                    # Only process if we have pose data. Otherwise, skip this image.
                    if self.last_gt_pose:
                        # Process image message.
                        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[:,:,:3] 
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                        # Create a filename for the image.
                        image_outpath = os.path.join(self.out_dir_images, str(self.image_counter).zfill(6) + ".png")
                        self.image_counter += 1
                        image_outpath_relative = "/".join(image_outpath.split("/")[-2:])
                        self.image_stamps.append(str(t.to_nsec()) +  " " + image_outpath_relative)
                        
                        # Save the image.
                        cv2.imwrite(image_outpath, img)

                        # Register the pose.
                        self.gt_pose_data.append(self.last_gt_pose)

                    # Set the flag to allow registering the next image.
                    self.allow_register_image = False
                    self.last_image_stamp = t.to_nsec()

                ##################
                # Process mocap.#
                ##################
                elif topic == self.mocap_topic:
                    # Process mocap message.

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
                    self.last_gt_pose=  [t.to_nsec(), dx, dy, dz, q[0], q[1], q[2], q[3]] 
            else:
                continue
            
            # If exceeded the timeframe requested, then break.
            if t.to_sec() - startt > self.bagend:
                break

        bag.close()

        # Save the image stamps.
        with open(self.out_file_image_stamps, "w") as f:    
            f.write("\n".join(self.image_stamps))
        
        # Save the pose data.
        self.gt_pose_data = np.vstack(self.gt_pose_data)
        np.savetxt(self.out_file_pose, self.gt_pose_data[:, 1:], fmt="%f %f %f %f %f %f %f")
        np.savetxt(self.out_file_pose_lcam_front, self.gt_pose_data[:, 1:], fmt="%f %f %f %f %f %f %f")

        # Save the original bagfile name.
        with open(os.path.join(self.out_dir_gp, "bagfile.txt"), "w") as f:
            f.write(self.bagfile_gp)

    def visualize_traj(self):
        # Visualize the trajectory.
        # Set a nice plt color scheme.
        plt.style.use('seaborn-whitegrid')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.gt_pose_data[:,1], self.gt_pose_data[:,2], self.gt_pose_data[:,3], label="GT Poses", color= 'black', linewidth=2)
        
        for ix, pose in enumerate(self.gt_pose_data):
            if ix % 10 != 0:
                continue
            pose_traj = self.gt_pose_data[ix]
            u, v, w = Rotation.from_quat(pose_traj[4:]).as_matrix() @ np.array([1, 0, 0])

        ax.legend()

        # Save to the output directory.
        plt.savefig(os.path.join(self.out_dir_gp, "traj3d.png"))

        # Save a 2D plot.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.gt_pose_data[:,1], self.gt_pose_data[:,2], label="GT Poses", color= 'black', linewidth=2)
        ax.legend()
        plt.savefig(os.path.join(self.out_dir_gp, "traj.png"))


    
def handle_args():
    # Set up command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bagfile', required=True, help='Path to the bagfile.')
    parser.add_argument('--image-topic', required=True, help='topic to extract images from.')
    parser.add_argument('-n', '--new-env-name', default='MocapEnv', help='Environment name given. Defaults to "MocapEnv"')
    parser.add_argument('--mocap-topic', required=True, help='topic to extract motion capture data from.')
    parser.add_argument('-o', '--tartanair-data-root', help='Path to the output directory.')
    parser.add_argument('-s', '--start', default='0')
    parser.add_argument('-e', '--end', default='2500')
    parser.add_argument('--hz', default='100', help='Frequency of the data in Hz.')
    parser.add_argument('--new-cam-name', default='lcam_fish', help='Name of the new camera.')
    
    return parser.parse_args()

def main():
    args = handle_args()
    bagfile_gp = args.bagfile

    # Check if the bagfile is a valid file or a directory.
    if os.path.isdir(bagfile_gp):
        bagfile_gp = [os.path.join(bagfile_gp, f) for f in os.listdir(bagfile_gp) if f.endswith(".bag")]
    elif os.path.isfile(bagfile_gp):
        bagfile_gp = [bagfile_gp]
    else:
        raise ValueError("Bagfile path is not a valid file or directory.")
    
    for bagfile in bagfile_gp:
        print(Fore.GREEN + f"Processing bagfile: {bagfile}" + Style.RESET_ALL)
        args.bagfile = bagfile

        b2f = BagToTartanAir(**vars(args))
        b2f.process_bag()
        b2f.visualize_traj()


if __name__ == "__main__":
    main()

'''
Example:
python3  bag_to_tartanair.py \
     -b /media/yoraish/overflow/data/fish_mocap/2023-05-09-11-38-29.bag \
     -o /media/yoraish/overflow/data/fish_mocap/tartanair_converted \
     --image-topic /camera_image1 \
     --mocap-topic /mocap_node/fish/pose \
     --hz 10
 

'''
