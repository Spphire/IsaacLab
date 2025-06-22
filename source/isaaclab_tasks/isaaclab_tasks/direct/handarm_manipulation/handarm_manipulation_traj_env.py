# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate, matrix_from_quat

if TYPE_CHECKING:
    from isaaclab_tasks.direct.shadow_hand_ur10e_arm.shadow_hand_ur10e_arm_env_cfg import ShadowHandUr10eArmEnvCfg

from isaaclab.utils.traj import BatchStateEstimator, LinearStateEstimator, RecordedTrajectory
CONTINUE_TRAJ = False

class HandArmManipulationTrajEnv(DirectRLEnv):
    cfg: ShadowHandUr10eArmEnvCfg

    def __init__(self, cfg: ShadowHandUr10eArmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        joint_pos_limits[:,:6,0] = self.hand.data.default_joint_pos[:,:6]-0.5
        joint_pos_limits[:,:6,1] = self.hand.data.default_joint_pos[:,:6]+0.5
        self.hand.root_physx_view.set_dof_limits(joint_pos_limits.cpu(), torch.arange(self.num_envs))

        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([0.7, 0.1, 0.3], device=self.device)
        self.goal_pos_init = self.goal_pos.clone()
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        
        self.forearm_link = self.hand.body_names.index('rh_forearm')
        self.palm_link = self.hand.body_names.index('rh_palm')
        
        self.play = self.num_envs<100
        if self.play:
            self.subepoch = self.cfg.subepoch_threshold
        else:
            self.subepoch = 0 
        self.keypoints_offset = torch.tensor([
                                                [ 1.0,  0.0,  0.0],  # right
                                                [-1.0,  0.0,  0.0],  # left
                                                [ 0.0,  1.0,  0.0],  # up
                                                [ 0.0, -1.0,  0.0],  # down
                                                [ 0.0,  0.0,  1.0],  # front
                                                [ 0.0,  0.0, -1.0],  # back
                                                [ 0.0,  0.0,  0.0],  # center
                                            ], dtype=torch.float, device=self.device)*self.cfg.keypoints_scale
        self.goal_keypoints = torch.zeros((self.num_envs, 7, 3), dtype=torch.float, device=self.device)
        self.object_keypoints = torch.zeros((self.num_envs, 7, 3), dtype=torch.float, device=self.device)

        self.duration = 1
        self.dt = self.cfg.sim.dt*self.cfg.decimation
        self.traj_len = int(self.duration/self.dt)
        self.traj_idx = torch.zeros((self.num_envs), dtype=torch.int, device=self.device)
        #self.traj_generator = RandomDivergingTrajectory(num_points=self.traj_len,dt=self.dt)
        self.traj_generator = RecordedTrajectory(traj_file="assets/physics_aware_interpolation.npy", dt=self.dt)
        self.traj_len = self.traj_generator.num_points
        self.obj_traj = torch.zeros((self.num_envs,self.traj_len,7+6+6), dtype=torch.float, device=self.device)
        self.obj_traj[...,3]=1.0

        self.robot_init = torch.zeros((self.num_envs,30+30), dtype=torch.float, device=self.device)

        #self.estimator = BatchStateEstimator(self.num_envs, dt = self.dt, device = self.device)
        self.estimator = LinearStateEstimator(self.num_envs, dt = self.dt, device = self.device)

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        if self.cfg.asymmetric_obs:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]

        obs = self.compute_full_observations()

        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
            self.subepoch,
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,

            self.object_keypoints,
            self.object_pos,
            self.object_rot,
            self.goal_keypoints,
            self.goal_pos,
            self.goal_rot,
            self.object_linacc-torch.Tensor([[ 0.0, 0.0, -9.81]]).repeat(self.num_envs, 1).to(self.device),
            self.forearm_rot-torch.Tensor([self.cfg.forearm_rot_target]).repeat(self.num_envs, 1).to(self.device),
            self.forearm_vel,

            self.cfg.forearm_linvel_error_threshold,
            self.cfg.forearm_rotvel_error_threshold,
            self.cfg.object_acc_diff_threshold,
            self.cfg.object_acc_penalty,
            
            self.cfg.forearm_linvel_error_weight,
            self.cfg.forearm_rotvel_error_weight,
            self.cfg.forearm_rot_error_weight,
            self.cfg.pos_error_weight,
            
            self.cfg.total_reward_scale,
            self.cfg.total_reward_eps,
            
            self.subepoch,
            self.cfg.subepoch_threshold,
            
            self.cfg.pos_success_threshold,
            self.cfg.forearm_rotvel_success_threshold,
            self.cfg.keypoints_success_threshold,
            
            self.cfg.reach_goal_bonus,
            self.cfg.fall_threshold,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0 and not CONTINUE_TRAJ:
            self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        out_of_reach = self.object_pos[...,2]<self.cfg.fall_threshold

        # if self.cfg.max_consecutive_success > 0:
        #     # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
        #     rot_dist = rotation_distance(self.object_rot, self.goal_rot)
        #     self.episode_length_buf = torch.where(
        #         torch.abs(rot_dist) <= self.cfg.success_tolerance,
        #         torch.zeros_like(self.episode_length_buf),
        #         self.episode_length_buf,
        #     )
        #     max_success_reached = self.successes >= self.cfg.max_consecutive_success

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # if self.cfg.max_consecutive_success > 0:
        #     time_out = time_out | max_success_reached
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        self.traj_idx[env_ids]=0

        # reset hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        rand_delta[:,:8] *= 0.05
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()
        
        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        
        object_default_state[:, 0:3]-=self.scene.env_origins[env_ids]
        self.obj_traj[env_ids] = self.traj_generator.generate(object_default_state[:, :7])
        self.estimator.reset(env_ids,object_default_state[:, :3],object_default_state[:, 3:7])
        self._reset_target_pose()

    def _reset_target_pose(self, env_ids = None):
        if not CONTINUE_TRAJ and env_ids is not None:
            self.traj_idx[env_ids] +=1
        env_ids = torch.arange(self.num_envs).to(self.device)
        self.goal_pos = self.obj_traj[env_ids,self.traj_idx,:3].view(-1,3)
        self.goal_rot = self.obj_traj[env_ids,self.traj_idx,3:7].view(-1,4)
        self.goal_linvel = self.obj_traj[env_ids,self.traj_idx,7:10].view(-1,3)
        self.goal_angvel = self.obj_traj[env_ids,self.traj_idx,10:13].view(-1,3)
        self.goal_linacc = self.obj_traj[env_ids,self.traj_idx,13:16].view(-1,3)
        self.goal_angacc = self.obj_traj[env_ids,self.traj_idx,16:19].view(-1,3)

        self.goal_pos1 = self.obj_traj[env_ids,self.traj_idx+1,:3].view(-1,3)
        self.goal_rot1 = self.obj_traj[env_ids,self.traj_idx+1,3:7].view(-1,4)
        self.goal_linvel1 = self.obj_traj[env_ids,self.traj_idx+1,7:10].view(-1,3)
        self.goal_angvel1 = self.obj_traj[env_ids,self.traj_idx+1,10:13].view(-1,3)
        self.goal_linacc1 = self.obj_traj[env_ids,self.traj_idx+1,13:16].view(-1,3)
        self.goal_angacc1 = self.obj_traj[env_ids,self.traj_idx+1,16:19].view(-1,3)

        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)
        self.goal_keypoints[env_ids] = transform_keypoints(self.keypoints_offset,self.goal_pos[env_ids],self.goal_rot[env_ids])


        self.reset_goal_buf[env_ids] = 0

    def _compute_intermediate_values(self, next_traj=True):
        # data for arm
        self.forearm_pos = self.hand.data.body_pos_w[:, self.forearm_link] - self.scene.env_origins
        self.forearm_rot = self.hand.data.body_quat_w[:, self.forearm_link]
        self.forearm_vel = self.hand.data.body_vel_w[:, self.forearm_link]
        self.forearm_acc = self.hand.data.body_acc_w[:, self.forearm_link]
        
        self.palm_pos = self.hand.data.body_pos_w[:, self.palm_link] - self.scene.env_origins
        self.palm_rot = self.hand.data.body_quat_w[:, self.palm_link]
        self.palm_vel = self.hand.data.body_vel_w[:, self.palm_link]   
        self.palm_acc = self.hand.data.body_acc_w[:, self.palm_link]
        
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w
        self.object_acc = self.object.data.body_acc_w
        self.object_linacc = self.object.data.body_acc_w[:,0,:3]
        self.object_angacc = self.object.data.body_acc_w[:,0,3:]
        self.object_keypoints = transform_keypoints(self.keypoints_offset,self.object_pos,self.object_rot)
        
        self.estimator.add_pose_sample(self.object_pos,self.object_rot)
        if CONTINUE_TRAJ:
            self._reset_target_pose()
            if next_traj:
                self.traj_idx[self.reward_state==1]+=1
        self.estimator_state = self.estimator.get_current_state()


    def compute_full_observations(self):
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                self.object_linacc,
                self.cfg.vel_obs_scale * self.object_angacc,
                # goal
                self.goal_pos,
                self.goal_rot,
                torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_full_state(self):
        states = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                self.object_linacc,
                self.cfg.vel_obs_scale * self.object_angacc,
                # goal
                self.goal_pos,
                self.goal_rot,
                torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return states


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


@torch.jit.script
def transform_keypoints(
    offsets: torch.Tensor,  # (7, 3)
    pos: torch.Tensor,      # (num_envs, 3)
    quat_wxyz: torch.Tensor # (num_envs, 4)
) -> torch.Tensor:
    """
    将固定偏移关键点变换到目标坐标系下
    返回: (num_envs, 7, 3)，目标关键点坐标
    """
    num_envs = pos.size(0)
    rot_mats = matrix_from_quat(quat_wxyz)  # (num_envs, 3, 3)
    
    offsets_expanded = offsets.unsqueeze(0).expand(num_envs, -1, -1)  # (num_envs, 7, 3)
    rotated_offsets = torch.bmm(rot_mats, offsets_expanded.transpose(1, 2)).transpose(1, 2)
    keypoints = rotated_offsets + pos.unsqueeze(1)  # (num_envs, 7, 3)
    return keypoints

@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    object_keypoints: torch.Tensor,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_keypoints: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    object_acc_diff: torch.Tensor,
    forearm_rot_diff: torch.Tensor,
    forearm_vel: torch.Tensor,
    forearm_linvel_error_threshold: float,
    forearm_rotvel_error_threshold: float,
    object_acc_diff_threshold: float,
    object_acc_penalty: float,
    
    forearm_linvel_error_weight: float,
    forearm_rotvel_error_weight: float,
    forearm_rot_error_weight: float,
    pos_error_weight: float,
    
    total_reward_scale: float,
    total_reward_eps: float,
    
    subepoch: int,
    subepoch_threshold: int,
    
    pos_success_threshold:float,
    forearm_rotvel_success_threshold:float,
    keypoints_success_threshold:float,
    
    reach_goal_bonus: float,
    fall_threshold: float,
    fall_penalty: float,
    av_factor: float,
):

    keypoints_error = torch.norm(object_keypoints - target_keypoints, p=2, dim=-1).mean(dim=1)
    pos_error = torch.norm(object_pos - target_pos, p=2, dim=-1)
    rot_error = torch.norm(object_rot - target_rot, p=2, dim=-1)
    forearm_linvel_error = torch.clamp(torch.norm(forearm_vel[:,:3], p=2, dim=-1)-forearm_linvel_error_threshold,min=0)
    forearm_rotvel_error = torch.clamp(torch.norm(forearm_vel[:,3:], p=2, dim=-1)-forearm_rotvel_error_threshold,min=0)
    forearm_rot_error = torch.norm(forearm_rot_diff, p=2, dim=-1)
    acc_diff_norm = torch.norm(object_acc_diff, p=2, dim=-1)

    # Total reward
    temp = forearm_linvel_error*forearm_linvel_error_weight+\
            forearm_rotvel_error*forearm_rotvel_error_weight+\
                forearm_rot_error*forearm_rot_error_weight+\
                    keypoints_error*pos_error_weight

    reward = - temp *total_reward_scale + 1/(temp+total_reward_eps)
    
    reward = torch.where((acc_diff_norm<object_acc_diff_threshold),reward-object_acc_penalty,reward)
    
    success_mask = (keypoints_error <= keypoints_success_threshold)

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(success_mask, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(object_pos[...,2]<fall_threshold, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(object_pos[...,2]<fall_threshold, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes, subepoch+1
