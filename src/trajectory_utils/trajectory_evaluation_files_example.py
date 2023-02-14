'''
Small script for evaluating the trajectory given files for an estimated trajectory and a ground truth trajectory.

The files should be in the format:
index tx ty tz qx qy qz qw

Author: Yorai Shaoul
Date: 2023-02-14 (Happy Valentine's Day!)
'''
# python imports
import os
import datetime

# Local imports.
from trajectory_evaluator_ate import TrajectoryEvaluatorATE
from trajectory_evaluator_rpe import TrajectoryEvaluatorRPE


####################
# Directories.
####################
est_traj_gp = '/media/yoraish/overflow/data/data/traj.txt'
gt_traj_gp  = '/media/yoraish/overflow/data/data/pose_lcam_front.txt'
output_dir = '/media/yoraish/overflow/data/data/output'
plot_ate_gfp = os.path.join(output_dir, 'ate.png')
plot_rpe_gfp = os.path.join(output_dir, 'rpe.png')

####################
# Verification of files.
####################
if not os.path.exists(output_dir):
    print("Creating output directory: {}".format(output_dir))
    os.makedirs(output_dir)


####################
# Evaluate the trajectory.
####################
# Compute the ATE.
ate_evaluator = TrajectoryEvaluatorATE(gt_file = gt_traj_gp, 
                                    est_file = est_traj_gp, 
                                    plot= True, 
                                    plot_gfp = plot_ate_gfp, 
                                    do_scale=True, 
                                    do_align=True)

ate, gt_traj, est_traj_aligned   = ate_evaluator.compute_ate(do_scale=True,
                                do_align=True)
print(f'---> ATE: {ate} m.')

rpe_evaluator = TrajectoryEvaluatorRPE(gt_file = gt_traj_gp,
                                        est_file = est_traj_gp,    
                                            plot= False,
                                            plot_gfp = plot_rpe_gfp,
                                            do_scale=True,
                                            do_align=True)

rpe, gt_traj, est_traj_aligned = rpe_evaluator.compute_rpe(do_scale=True,
                                do_align=True)
print(f'---> RPE: {rpe}.')

timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
plot_ate_gfp = os.path.join(output_dir, 'ate_{}.png'.format(timenow))
plot_rpe_gfp = os.path.join(output_dir, 'rpe_{}.png'.format(timenow))


