'''
Adapted by: Yorai Shaoul 
Date: December 2022

# Modified by Wenshan Wang
# Modified by Raul Mur-Artal
# Automatically compute the optimal scale factor for monocular VO/SLAM.

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

Compute the absolute trajectory error (ATE) for a monocular VO/SLAM system. ATE is computed as the mean of the translational error over all frames. The translational error is computed as the Euclidean distance between the ground truth and the estimated trajectory. Both the ground truth trajectory and the estimatedd trajectory are specified in the same reference frame. 

The input trajectories are expected to be in the form:
    stamp0 tx0 ty0 tz0 qx0 qy0 qz0 qw0 
    stamp1 tx1 ty1 tz1 qx1 qy1 qz1 qw1
    stamp2 tx2 ty2 tz2 qx2 qy2 qz2 qw2
    ...

where stamp is the ROS timestamp, tx ty tz is the translation, and qx qy qz qw is the quaternion representing the rotation. Stamp is expected to be in nanoseconds, but can also be an integer index. Correspondence between the ground truth and the estimated trajectory is established by matching the stamps.

A single scale factor is computed for the entire trajectory, and can alternatively be provided.

Run with a command like:

python3 avaluate_ate_scale.py   --gt-file /home/user/data/gt_traj.txt \
                                --est-file /home/user/data/est_traj.txt

If you do not have a trajectory file, you can generate one with the following command (the file is also in this repository):

python3 bag_to_traj_dataset.py  -b /home/path/to/bagfile.bag \
                                --image-topic /camera_0/image \
                                --imu-topic /imu/data \
                                --mocap-topic /mocap_node/robot/pose \
                                -o /path/to/output/directory \
                                -s starttime \
                                -e endtime
'''

# General imports.
from colorama import Fore, Style
import argparse
import numpy as np
import os
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from matplotlib import pyplot as plt


class TrajATEScaleEvaluator(object):
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gt-file', required=True, help='Path to the ground truth trajectory file.')
        parser.add_argument('--est-file', required=True, help='Path to the estimated trajectory file.')
        parser.add_argument('--plot', action='store_true', help='Plot the results.')
        parser.add_argument('--plot-dir', default='/home', help='Path to the directory where the plot will be saved.')
        parser.add_argument('--no-calc-scale', action='store_true', help="If flag existing, unity scale is applied to the estimated trajectory, and not computed, before computing the ATE.")
        parser.add_argument('--verbose', action='store_true', help='Print the results.')
        return parser.parse_args()

    def __init__(self):
        
        #####################
        # Parse the command line arguments
        # and set parameters.
        #####################
        # Set up command line arguments.
        self.args = self.parse_args()

        # Set up the ground truth and estimated trajectory file paths.
        self.gt_file = self.args.gt_file
        self.est_file = self.args.est_file

        # Set up the plot directory.
        self.plot_dir = self.args.plot_dir

        # Set up a constant scale factor, if one is provided.
        if self.args.no_calc_scale:
            self.calc_scale = False
        else: 
            self.calc_scale = True
            
        
        # Set up the verbose flag.
        self.verbose = self.args.verbose

        # Set up the plot flag.
        self.plot = self.args.plot

        # Read the trajectories.
        self.gt_traj = self.read_trajectory(self.gt_file)
        self.est_traj = self.read_trajectory(self.est_file)

        # Placeholder for estimated trajectory aligned to the ground truth and scaled.
        self.est_traj_aligned = None

    def read_trajectory(self, traj_file):
        """Read a trajectory from a file.
        Args:
            traj_file (str): Path to the trajectory file.
        """
        traj = np.loadtxt(traj_file)
        return traj


    def compute_ate(self):
        """Compute the absolute trajectory error (ATE).
            self.gt_traj (np.ndarray): Ground truth trajectory.
            self.est_traj (np.ndarray): Estimated trajectory.

            returns:
                ate (float): Absolute trajectory error.

        """
        # Get the xyz positions from the ground truth and estimated trajectories.
        gt_xyz = self.gt_traj[:, 1:4].T # 3 x N
        est_xyz = self.est_traj[:, 1:4].T # 3 x N

        # Align the estimated trajectory to the ground truth trajectory, get the alignment transformation, and the scale if requested.
        R_gt_in_est, t_gt_in_est, trans_error, s = self.align(gt_xyz, est_xyz, calc_scale=self.calc_scale)

        # Compute the ATE.
        # TODO(yoraish): I don't know if we need the square and then sqrt here. The output of the align function is already the norm of the error. We may just need to average here.
        ate = np.sqrt(np.dot(trans_error,trans_error) / len(trans_error))

        # Go back to the provided estimated trajectory and apply the scale factor and alignment transformation to it.
        # This rotates the provided quaternions as well.
        # align two trajs 
        est_SEs = self.pos_quats_to_SE_matrices(self.est_traj[:,1:]) # Pass in the estimated trajectory, without the timestamps.

        gt_in_est = np.eye(4) 
        gt_in_est[:3,:3] = R_gt_in_est
        gt_in_est[:3,3:] = t_gt_in_est 
        est_in_gt = np.linalg.inv(gt_in_est)
        est_traj_aligned = []

        for se in est_SEs:
            se[:3,3] = se[:3,3] * s
            se_new = est_in_gt.dot(se)
            se_new = self.SE_to_pos_quat(se_new)
            est_traj_aligned.append(se_new)

        est_traj_aligned = np.array(est_traj_aligned)
        self.est_traj_aligned = np.hstack( (self.est_traj[:,0:1] , est_traj_aligned) )
        self.ate = ate
        # Return the ATE, the ground truth trajectory without its timestamps, and the aligned estimated trajectory.
        return ate, self.gt_traj[:, 1:], est_traj_aligned


    ##########################
    # Transformation functions.
    ##########################
    def SE_to_pos_quat(self, SE):
        """Convert a 4x4 SE matrix to a 7-vector of position and quaternion. xyz, xyzw.
        """
        pos = SE[:3,3]
        quat = Rotation.from_matrix(SE[:3,:3]).as_quat()
        return np.concatenate([pos, quat])

    def pos_quats_to_SE_matrices(self, pos_quats):
        """Convert a list of 7-vectors of position and quaternion to a list of 4x4 SE matrices.
        """
        SEs = []
        for pos_quat in pos_quats:
            pos = pos_quat[:3]
            quat = pos_quat[3:]
            SE = np.eye(4)
            SE[:3,:3] = Rotation.from_quat(quat).as_matrix()
            SE[:3,3] = pos
            SEs.append(SE)
        return SEs

    def align(self, gt_xyz, est_xyz, calc_scale=False):
        """Align two trajectories using the method of Horn (closed-form).
        
        Input:
        model -- first trajectory (3xn)
        est_xyz -- second trajectory (3xn)
        
        Output:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)
        
        """
        np.set_printoptions(precision=3,suppress=True)
        gt_xyz_zerocentered = gt_xyz - gt_xyz.mean(1).reshape((3,1))
        est_xyz_zerocentered = est_xyz - est_xyz.mean(1).reshape((3,1))
        
        W = np.zeros( (3,3) )
        for column in range(gt_xyz.shape[1]):
            W += np.outer(gt_xyz_zerocentered[:,column],est_xyz_zerocentered[:,column])
        U,d,Vh = np.linalg.linalg.svd(W.transpose())
        S = np.matrix(np.identity( 3 ))
        if(np.linalg.det(U) * np.linalg.det(Vh)<0):
            S[2,2] = -1
        rot = U*S*Vh

        if calc_scale:
            rotgt_xyz = rot*gt_xyz_zerocentered
            dots = 0.0
            norms = 0.0
            for column in range(est_xyz_zerocentered.shape[1]):
                dots += np.dot(est_xyz_zerocentered[:,column].transpose(),rotgt_xyz[:,column])
                normi = np.linalg.norm(gt_xyz_zerocentered[:,column])
                norms += normi*normi
            s = float(norms/dots)
        else:
            s = 1.0  

        # Scale the est to the gt, otherwise the ATE could be very small if the est scale is small
        trans = s * est_xyz.mean(1).reshape((3,1)) - rot @ gt_xyz.mean(1).reshape((3,1))
        gt_xyz_aligned = rot @ gt_xyz + trans
        est_xyz_alingned = s * est_xyz
        alignment_error = gt_xyz_aligned - est_xyz_alingned
        
        trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]
            
        return rot,trans,trans_error, s


    ##########################
    # Visualization functions.
    ##########################
    def visualize(self):
        """Visualize the ground truth trajectory and the estimated trajectory.
        """
        # Visualize the trajectory.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.gt_traj[:,1], self.gt_traj[:,2], self.gt_traj[:,3], label="GT", color = 'k')
        if self.est_traj_aligned is not None:
            ax.plot(self.est_traj_aligned[:,1], self.est_traj_aligned[:,2], self.est_traj_aligned[:,3], label="EST. Aligned and scaled.", color = 'r', alpha=0.5)
        
        for ix, pose in enumerate(self.gt_traj):
            if ix % 50 != 0:
                continue
            pose_gt = self.gt_traj[ix]
            u, v, w = Rotation.from_quat(pose_gt[4:]).as_matrix() @ np.array([1, 0, 0])
            ax.quiver(pose_gt[1], pose_gt[2], pose_gt[3], u, v, w, length=0.1, normalize=True, color="blue", alpha=0.5)

            if self.est_traj_aligned is not None:
                pose_est = self.est_traj_aligned[ix]
                u, v, w = Rotation.from_quat(pose_est[4:]).as_matrix() @ np.array([1, 0, 0])
                ax.quiver(pose_est[1], pose_est[2], pose_est[3], u, v, w, length=0.1, normalize=True, color="red", alpha = 0.5)
            
        ax.legend()
        ax.set_title("ATE: {:.3f} m".format(self.ate))
        plt.show()

    def visualize_2d_projection(self):
        """Visualize the ground truth trajectory and the estimated trajectory.
        """
        # Visualize the trajectory.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.gt_traj[:,1], self.gt_traj[:,2], label="GT", color = 'k')
        if self.est_traj_aligned is not None:
            ax.plot(self.est_traj_aligned[:,1], self.est_traj_aligned[:,2], label="EST. Aligned and scaled.", color = 'r', alpha=0.5)
        
        for ix, pose in enumerate(self.gt_traj):
            if ix % 50 != 0:
                continue
            pose_gt = self.gt_traj[ix]
            u, v, w = Rotation.from_quat(pose_gt[4:]).as_matrix() @ np.array([0.1, 0, 0])
            ax.quiver(pose_gt[1], pose_gt[2], u, v, color="blue", alpha= 0.2)

            if self.est_traj_aligned is not None:
                pose_est = self.est_traj_aligned[ix]
                u, v, w = Rotation.from_quat(pose_est[4:]).as_matrix() @ np.array([1, 0, 0])
                ax.quiver(pose_est[1], pose_est[2], u, v, color="red", alpha = 0.2)
        
        ax.set_title("ATE: {:.3f} m".format(self.ate))
        ax.legend()
        plt.show()

if __name__ == "__main__":
    ate_evaluator = TrajATEScaleEvaluator()
    ate, gt_traj, est_traj_aligned = ate_evaluator.compute_ate()
    ate_evaluator.visualize_2d_projection()
