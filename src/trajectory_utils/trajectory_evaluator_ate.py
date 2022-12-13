'''
Adapted by: Yorai Shaoul 
Date: December 2022

Modified by Wenshan Wang
Modified by Raul Mur-Artal
Automatically compute the optimal scale factor for monocular VO/SLAM.

Software License Agreement (BSD License)

Copyright (c) 2013, Juergen Sturm, TUM
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.
 * Neither the name of TUM nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Compute the absolute trajectory error (ATE) for a monocular VO/SLAM system. ATE is computed as the mean of the translational error over all frames. The translational error is computed as the Euclidean distance between the ground truth and the estimated trajectory. Both the ground truth trajectory and the estimatedd trajectory are specified in the same reference frame. 

The input trajectories are expected to be in the form:
    stamp0 tx0 ty0 tz0 qx0 qy0 qz0 qw0 
    stamp1 tx1 ty1 tz1 qx1 qy1 qz1 qw1
    stamp2 tx2 ty2 tz2 qx2 qy2 qz2 qw2
    ...

where stamp is the ROS timestamp, tx ty tz is the translation, and qx qy qz qw is the quaternion representing the rotation. Stamp is expected to be in nanoseconds, but can also be an integer index. Correspondence between the ground truth and the estimated trajectory is established by the order of the poses in the trajectory.

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

# Local imports.
from trajectory_evaluator_base import TrajectoryEvaluatorBase

class TrajectoryEvaluatorATE(TrajectoryEvaluatorBase):
    def parse_args(self):
        
        # Instantiate super class.
        super(TrajectoryEvaluatorATE, self)

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
            self.do_scale = False
        else: 
            self.do_scale = True
            
        
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


    def compute_rpe(self, gt_traj = None, est_traj = None, do_scale=None, do_align=False):
        """Compute the absolute trajectory error (ATE).
            self.gt_traj (np.ndarray , (N,8): stamp, xyz xyzw): Ground truth trajectory.
            self.est_traj (np.ndarray, (N,8): stamp, xyz xyzw): Estimated trajectory.

            returns:
                ate (float): Absolute trajectory error.

        """
        # Use the member variables if an input is not provided.
        if gt_traj is None:
            gt_traj = self.gt_traj
        if est_traj is None:
            est_traj = self.est_traj
        if do_scale is None:
            do_scale = self.do_scale
    

        # Align the estimated trajectory to the ground truth trajectory, get the alignment transformation, and the scale if requested.
        if do_align:
            est_traj_aligned, s = self.align_and_scale_traj_to_gt(gt_traj, est_traj, do_scale)
            print(Fore.GREEN + "Scale factor: " + str(s) + Style.RESET_ALL)
        else:
            est_traj_aligned = est_traj
            s = 1.0
        
        self.est_traj_aligned = est_traj_aligned
        ate = np.sqrt( (np.sum( np.linalg.norm(self.est_traj_aligned[:, 1:4] - gt_traj[:, 1:4], axis = 1)**2 ) / len(est_traj_aligned)  ) )
        self.ate = ate

        if self.plot:
            title_text = "ATE: {:.3f} m".format(ate)
            self.visualize(gt_traj, est_traj_aligned, title_text)
            self.visualize_2d_projection(gt_traj, est_traj_aligned, title_text)

        # Return the ATE, the ground truth trajectory with its timestamps, and the aligned estimated trajectory.
        return ate, gt_traj, est_traj_aligned



if __name__ == "__main__":
    ate_evaluator = TrajectoryEvaluatorATE()
    ate, gt_traj, est_traj_aligned = ate_evaluator.compute_ate()
    print(Fore.GREEN + "ATE: " + str(ate) + Style.RESET_ALL)
    print(Fore.GREEN + "GT and aligned estimated trajectory shapes: " + str(gt_traj.shape) + " " + str(est_traj_aligned.shape) + Style.RESET_ALL)