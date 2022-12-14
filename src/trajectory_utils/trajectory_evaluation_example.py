"""
This script is an example of how to use the trajectory evaluation functions using python alone, without command line arguments.
"""

import numpy as np
from trajectory_evaluator_rpe import TrajectoryEvaluatorRPE
from trajectory_evaluator_ate import TrajectoryEvaluatorATE

# Set up the ground truth and estimated trajectories.
# Each one is a numpy array of shape (N, 8) where N is the number of poses. Taking the format:
# [timestamp, x, y, z, qx, qy, qz, qw]

# Ground truth trajectory.
gt_traj = np.zeros((100,8))
gt_traj[:,0] = np.arange(0,100)
gt_traj[:,1] = np.arange(0,100)
gt_traj[:,2] = np.arange(0,100)
gt_traj[:,3] = np.arange(0,100) **2
gt_traj[:,4:7] = np.zeros((100, 3))
gt_traj[:,7] = np.ones(100)

# Estimated trajectory.
est_traj = np.zeros((100,8))
est_traj[:, 1] = np.arange(0,100) * 0.5 + np.random.normal(1, 0.5, 100)  
est_traj[:, 2] = np.arange(0,100) * 0.5 + np.random.normal(1, 0.1, 100) 
est_traj[:, 3] = np.arange(0,100) ** 2 * 0.5 + np.random.normal(1, 0.1, 100) 
est_traj[:,4:7] = np.zeros((100, 3))
est_traj[:, 7] = np.ones(100)

# Create the trajectory evaluators.
rpe_evaluator = TrajectoryEvaluatorRPE()
ate_evaluator = TrajectoryEvaluatorATE()

# Compute the errors.
rpe, gt_traj, est_traj_aligned   = rpe_evaluator.compute_rpe(gt_traj, est_traj, delta = 1, do_scale = False, do_align = False)
trans_error_mean, rot_error_mean = rpe
title_text = "RPE " + "{:.5f}".format(trans_error_mean) + " m, " + "{:.5f}".format(rot_error_mean) + " radians(?)"
rpe_evaluator.visualize(gt_traj, est_traj_aligned, title_text, arrow_length=10)


ate, gt_traj, est_traj_aligned   = ate_evaluator.compute_ate(gt_traj, est_traj, do_scale = True, do_align = True)

# Plot the errors.
title_text = "ATE: {:.3f} m".format(ate)
ate_evaluator.visualize(gt_traj, est_traj_aligned, title_text, arrow_length=10)


# Print the errors.
print("RPE errors: ", rpe)
print("ATE errors: ", ate)