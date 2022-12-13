import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

class TrajectoryEvaluatorBase (): 
    """Base class for trajectory evaluators."""
    
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


    def ominus(self, a, b):
        """Compute the relative pose error.
        Args:
            a (np.array, 4x4): First pose.
            b (np.array, 4x4): Second pose.
        """
        a_inv = np.linalg.inv(a)
        return np.dot(a_inv, b)

    def compute_distance(self, transform):
        """Compute the translation error.
        Args:
            transform (np.array, 4x4): Relative pose error.
        """
        return np.linalg.norm(transform[0:3,3])    


    def compute_angle(self, transform):
        """
        Compute the rotation angle from a 4x4 homogeneous matrix.
        """
        # an invitation to 3-d vision, p 27
        return np.arccos( min(1,max(-1, (np.trace(transform[0:3,0:3]) - 1)/2) ))


    ##########################
    # Visualization functions.
    ##########################
    def visualize(self, gt_traj, est_traj, title_text = ''):
        """Visualize the ground truth trajectory and the estimated trajectory.
        """
        # Visualize the trajectory.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(gt_traj[:,1], gt_traj[:,2], gt_traj[:,3], label="GT", color = 'k')
        if est_traj is not None:
            ax.plot(est_traj[:,1], est_traj[:,2], est_traj[:,3], label="EST. Aligned and scaled.", color = 'r', alpha=0.5)
        
        for ix, pose in enumerate(gt_traj):
            if ix % 50 != 0:
                continue
            pose_gt = gt_traj[ix]
            u, v, w = Rotation.from_quat(pose_gt[4:]).as_matrix() @ np.array([1, 0, 0])
            ax.quiver(pose_gt[1], pose_gt[2], pose_gt[3], u, v, w, length=0.1, normalize=True, color="blue", alpha=0.5)

            if est_traj is not None:
                pose_est = est_traj[ix]
                u, v, w = Rotation.from_quat(pose_est[4:]).as_matrix() @ np.array([1, 0, 0])
                ax.quiver(pose_est[1], pose_est[2], pose_est[3], u, v, w, length=0.1, normalize=True, color="red", alpha = 0.5)
            
        ax.legend()
        ax.set_title(title_text)
        plt.show()

    def visualize_2d_projection(self, gt_traj, est_traj, title_text = ''):
        """Visualize the ground truth trajectory and the estimated trajectory.
        """
        # Visualize the trajectory.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(gt_traj[:,1], gt_traj[:,2], label="GT", color = 'k')
        if est_traj is not None:
            ax.plot(est_traj[:,1], est_traj[:,2], label="EST. Aligned and scaled.", color = 'r', alpha=0.5)
        
        for ix, pose in enumerate(gt_traj):
            if ix % 50 != 0:
                continue
            pose_gt = gt_traj[ix]
            u, v, w = Rotation.from_quat(pose_gt[4:]).as_matrix() @ np.array([0.1, 0, 0])
            ax.quiver(pose_gt[1], pose_gt[2], u, v, color="blue", alpha= 0.2)

            if est_traj is not None:
                pose_est = est_traj[ix]
                u, v, w = Rotation.from_quat(pose_est[4:]).as_matrix() @ np.array([1, 0, 0])
                ax.quiver(pose_est[1], pose_est[2], u, v, color="red", alpha = 0.2)
        
        ax.set_title(title_text)
        ax.legend()
        plt.show()

    def __init__(self, gt_traj, est_traj):
        """Initialize the trajectory evaluator.
        
        Input:
        gt_traj -- ground truth trajectory (8xn) i, xyz, xyzw.
        est_traj -- estimated trajectory (8xn) i, xyz, xyzw.
        max_diff -- maximum difference between timestamps to consider a match.
        
        """
        self.gt_traj = gt_traj
        self.est_traj = est_traj

    def align_and_scale_traj_to_gt(self, gt_traj, est_traj, calc_scale=False):
        """Align two trajectories using the method of Horn (closed-form).
        
        Input:
        model -- first trajectory (8xn) i, xyz, xyzw.
        est_xyz -- second trajectory (8xn) i, xyz, xyzw.
        
        Output:
        est_traj_aligned -- second trajectory aligned to the first (8xn) i, xyz, xyzw.
        scale -- scale factor
        
        """
        np.set_printoptions(precision=3,suppress=True)

        # Get the xyz positions from the ground truth and estimated trajectories.
        gt_xyz = gt_traj[:, 1:4].T # 3 x N
        est_xyz = est_traj[:, 1:4].T # 3 x N
        
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
        est_xyz_aligned = s * est_xyz
        alignment_error = gt_xyz_aligned - est_xyz_aligned
        trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]
        # Option to compute the ATE directly from the alignment error.
        # ate = np.sqrt(np.dot(trans_error,trans_error) / len(trans_error))

        # Create the transformed trajectory. This rotates the provided quaternions as well.
        est_SEs = self.pos_quats_to_SE_matrices(self.est_traj[:,1:]) # Pass in the estimated trajectory, without the timestamps.

        R_gt_in_est, t_gt_in_est = rot, trans
        gt_in_est = np.eye(4) 
        gt_in_est[:3,:3] = R_gt_in_est
        gt_in_est[:3,3:] = t_gt_in_est 
        est_in_gt = np.linalg.inv(gt_in_est)
        est_traj_aligned = []

        # Transform the estimated trajectory.
        for ix, se in enumerate(est_SEs):
            se[:3,3] = se[:3,3] * s
            se_new = est_in_gt.dot(se)
            se_new = self.SE_to_pos_quat(se_new)
            est_traj_aligned.append(se_new)

        # Convert to numpy array.
        est_traj_aligned = np.array(est_traj_aligned)
        
        # Add the timestamps back in.
        est_traj_aligned = np.hstack( (est_traj[:,0:1] , est_traj_aligned) )

        return est_traj_aligned, s