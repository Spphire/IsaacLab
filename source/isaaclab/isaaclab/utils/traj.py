import torch
from numpy import load as npload
from isaaclab.utils.math import quat_mul, quat_inv

def quat_to_angular_velocity(q1, q2, dt):
    """从相对旋转四元数计算角速度 (wxyz 格式)"""
    q_rel = quat_mul(q2, quat_inv(q1))
    x, y, z = q_rel[..., 1], q_rel[..., 2], q_rel[..., 3]
    return (2 / dt) * torch.stack([x, y, z], dim=-1)

class RandomTrajectoryV2:
    def __init__(self, num_points, dt=1/60,
                 max_lin_speed=0.2, max_lin_accel=1.0,
                 max_ang_speed=5.0, max_ang_accel=5.0, device='cuda:0'):
        self.num_points = num_points
        self.dt = dt
        self.max_lin_speed = max_lin_speed
        self.max_lin_accel = max_lin_accel
        self.max_ang_speed = max_ang_speed
        self.max_ang_accel = max_ang_accel
        self.device = device
        self.traj_points = None

    def generate(self, initial_pose):
        batch_size = initial_pose.shape[0]
        traj = torch.zeros((batch_size, self.num_points, 19),
                           dtype=torch.float32, device=self.device)

        traj[:, 0, :3] = initial_pose[:, :3]
        traj[:, 0, 3:7] = initial_pose[:, 3:7]

        current_lin_vel = torch.zeros((batch_size, 3), device=self.device)
        current_lin_acc = torch.zeros((batch_size, 3), device=self.device)
        current_ang_vel = torch.zeros((batch_size, 3), device=self.device)
        current_ang_acc = torch.zeros((batch_size, 3), device=self.device)

        change_indices = [
            sorted((torch.randperm(self.num_points - 20)[:torch.randint(1, 4, ()).item()] + 10).tolist())
            for _ in range(batch_size)
        ]

        for i in range(1, self.num_points):
            # 在扰动点注入强制方向变更
            for b in range(batch_size):
                if i in change_indices[b]:
                    # === 强制线速度偏转 ===
                    prev_dir = current_lin_vel[b] / (torch.norm(current_lin_vel[b]) + 1e-6)
                    rand_dir = torch.randn(3, device=self.device)
                    rand_dir = rand_dir - torch.dot(rand_dir, prev_dir) * prev_dir  # 与原方向正交
                    rand_dir = rand_dir / (torch.norm(rand_dir) + 1e-6)
                    desired_dir = (prev_dir + 2.0 * rand_dir)
                    desired_dir = desired_dir / (torch.norm(desired_dir) + 1e-6)
                    current_lin_acc[b] = desired_dir * self.max_lin_accel * 1.5  # 强化加速

                    # === 强制角速度突变 ===
                    current_ang_acc[b] = torch.randn(3, device=self.device) * self.max_ang_accel * 2.0

            current_lin_acc = 0.7 * current_lin_acc + 0.3 * (
                (torch.rand((batch_size, 3), device=self.device) - 0.5) * 2 * self.max_lin_accel)
            current_lin_acc = torch.clamp(current_lin_acc, -self.max_lin_accel, self.max_lin_accel)

            prev_lin_vel = current_lin_vel.clone()
            current_lin_vel += current_lin_acc * self.dt
            lin_speed = torch.norm(current_lin_vel, dim=1, keepdim=True)
            exceed = lin_speed > self.max_lin_speed
            current_lin_vel = torch.where(exceed, current_lin_vel / (lin_speed + 1e-6) * self.max_lin_speed, current_lin_vel)
            current_lin_acc = (current_lin_vel - prev_lin_vel) / self.dt

            traj[:, i, :3] = traj[:, i - 1, :3] + current_lin_vel * self.dt

            current_ang_acc = 0.8 * current_ang_acc + 0.2 * (
                torch.randn((batch_size, 3), device=self.device) * self.max_ang_accel)
            current_ang_acc = torch.clamp(current_ang_acc, -self.max_ang_accel, self.max_ang_accel)

            prev_ang_vel = current_ang_vel.clone()
            current_ang_vel += current_ang_acc * self.dt
            ang_speed = torch.norm(current_ang_vel, dim=1, keepdim=True)
            current_ang_vel = torch.where(ang_speed > self.max_ang_speed,
                                          current_ang_vel / (ang_speed + 1e-6) * self.max_ang_speed,
                                          current_ang_vel)
            current_ang_acc = (current_ang_vel - prev_ang_vel) / self.dt

            angle = torch.norm(current_ang_vel, dim=1) * self.dt
            axis = current_ang_vel / (ang_speed + 1e-8)
            half_angle = angle * 0.5
            sin_h = torch.sin(half_angle)
            delta_quat = torch.cat([torch.cos(half_angle).unsqueeze(1),
                                    axis * sin_h.unsqueeze(1)], dim=1)

            prev_quat = traj[:, i - 1, 3:7]
            new_quat = quat_mul(prev_quat, delta_quat)
            new_quat = new_quat / (torch.norm(new_quat, dim=1, keepdim=True) + 1e-8)
            traj[:, i, 3:7] = new_quat

            traj[:, i - 1, 7:10] = current_lin_vel
            traj[:, i - 1, 10:13] = current_ang_vel
            traj[:, i - 1, 13:16] = current_lin_acc
            traj[:, i - 1, 16:19] = current_ang_acc

        traj[:, -1, 7:10] = current_lin_vel
        traj[:, -1, 10:13] = current_ang_vel
        traj[:, -1, 13:16] = current_lin_acc
        traj[:, -1, 16:19] = current_ang_acc

        self.traj_points = traj
        return traj

class RecordedTrajectory:
    def __init__(self, traj_file: str, dt:float, device: str = "cuda:0", random_rate=0):
        """
        从文件加载预录制的轨迹数据
        
        参数:
            traj_file: 轨迹文件路径
            device: 计算设备
        """
        self.dt=dt
        self.device = device
        self.random_rate = random_rate
        #load from .npy
        if traj_file.endswith('.npy'):
            self.traj_data = torch.tensor(npload(traj_file), dtype=torch.float32, device=device)
        else:
            raise ValueError("Unsupported trajectory file format. Only .npy files are supported.")
        self.traj_data = self.traj_data[:,:62,:]
        self.num_points = self.traj_data.shape[1]
        
        self.traj_data[:,:,:3] = self.traj_data[:,:, :3] - self.traj_data[:, [0], :3].clone()  # 减去初始位置
        #减去初始旋转
        self.traj_data[:, :, 3:7] = quat_mul(
            quat_inv(self.traj_data[:, [0], 3:7].clone()).repeat(1, self.num_points, 1),
            self.traj_data[:, :, 3:7]
        )
        self.random = RandomTrajectoryV2(num_points=self.num_points, dt=self.dt)
    
    def generate(self, initial_pose: torch.Tensor):
        """
        生成轨迹数据
        
        参数:
            initial_pose: 初始位置和姿态 (batch_size, 7) [x, y, z, qw qx, qy, qz]
        
        返回:
            轨迹数据 (batch_size, num_points, 19)
        """
        random_traj = self.random.generate(initial_pose)

        batch_size = initial_pose.shape[0]
        #random sort traj_data
        indices = torch.randperm(self.traj_data.shape[0], device=self.device)
        self.traj_data = self.traj_data[indices]
        if self.traj_data.shape[0] < batch_size:
            repeat_times = (batch_size + self.traj_data.shape[0] - 1) // self.traj_data.shape[0]
            traj_data_expanded = self.traj_data.repeat(repeat_times, 1, 1).clone()
            traj_data_expanded = traj_data_expanded[:batch_size]
        else:
            traj_data_expanded = self.traj_data[:batch_size].clone()
        
        # 将初始位置和姿态应用到轨迹数据
        initial_position = initial_pose[:, :3].unsqueeze(1)  # (batch_size, 1, 3)
        initial_quat = initial_pose[:, 3:7].unsqueeze(1)  # (batch_size, 1, 4)
        traj_data_expanded[:, :, :3] += initial_position  # 添加初始位置
        traj_data_expanded[:, :, 3:7] = quat_mul(initial_quat, traj_data_expanded[:, :, 3:7])  # 应用初始旋转
        # calculate and 添加速度和加速度
        traj_data_expanded = traj_data_expanded.to(self.device)
        traj_data_expanded = traj_data_expanded.clone()  # 确保不修改原始数据
        # 计算线速度和角速度
        linvel = (traj_data_expanded[:, 1:, :3] - traj_data_expanded[:, :-1, :3]) / self.dt
        linvel =torch.cat(
            (
                torch.zeros((linvel.shape[0],1,3)).to(self.device),
                linvel,
            ),
            dim = -2
        )
        angvel = quat_to_angular_velocity(
            traj_data_expanded[:, :-1, 3:7],
            traj_data_expanded[:, 1:, 3:7],
            self.dt
        )
        angvel =torch.cat(
            (
                torch.zeros((angvel.shape[0],1,3)).to(self.device),
                angvel,
            ),
            dim = -2
        )
        # 计算线加速度和角加速度
        linacc = (linvel[:, 1:, :] - linvel[:, :-1, :]) / self.dt
        linacc =torch.cat(
            (
                torch.zeros((linacc.shape[0],1,3)).to(self.device),
                linacc,
            ),
            dim = -2
        )
        angacc = (angvel[:, 1:, :] - angvel[:, :-1, :]) / self.dt
        angacc =torch.cat(
            (
                torch.zeros((angacc.shape[0],1,3)).to(self.device),
                angacc,
            ),
            dim = -2
        )
        # 将速度和加速度添加到轨迹数据中
        traj_data_expanded = torch.cat([
            traj_data_expanded,
            linvel,
            angvel,
            linacc,
            angacc
        ], dim=-1)
        return torch.where(torch.rand(batch_size,1,1).to('cuda')<self.random_rate,random_traj,traj_data_expanded)
    
class LinearStateEstimator:
    def __init__(self, num_envs, dt=1/60, device="cuda"):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        self.position_history = torch.zeros((num_envs, 2, 3), dtype=torch.float32, device=device)
        self.quat_history = torch.zeros((num_envs, 2, 4), dtype=torch.float32, device=device)

        self.linvel = torch.zeros((num_envs,2, 3), device=device)
        self.angvel = torch.zeros((num_envs,2, 3), device=device)

        self.t = torch.zeros((num_envs,2), device=device) * dt

        self.linacc = torch.zeros((num_envs, 3), device=device)
        self.angacc = torch.zeros((num_envs, 3), device=device)

    def reset(self, env_ids, init_pos: torch.Tensor, init_quat: torch.Tensor):
        assert init_pos.shape[0] == env_ids.shape[0], "init_pos should match env_ids in length"
        assert init_quat.shape[0] == env_ids.shape[0], "init_quat should match env_ids in length"

        self.position_history[env_ids] = init_pos.unsqueeze(1).expand(-1, 2, -1)
        self.quat_history[env_ids] = init_quat.unsqueeze(1).expand(-1, 2, -1)

        self.linvel[env_ids] = 0
        self.angvel[env_ids] = 0
        self.linacc[env_ids] = 0
        self.angacc[env_ids] = 0

        self.t[env_ids] = 0

    def add_pose_sample(self, position, quat_wxyz):
        self.position_history = torch.roll(self.position_history, shifts=-1, dims=1)
        self.quat_history = torch.roll(self.quat_history, shifts=-1, dims=1)
        self.position_history[:, -1] = position
        self.quat_history[:, -1] = quat_wxyz

        self.linvel = torch.roll(self.linvel, shifts=-1, dims=1)
        self.angvel = torch.roll(self.angvel, shifts=-1, dims=1)
        self.linvel[:, -1] = (self.position_history[:, -1] - self.position_history[:, -2]) / self.dt
        self.angvel[:, -1] = quat_to_angular_velocity(self.quat_history[:, -2], self.quat_history[:, -1], self.dt)

        self.linacc = (self.linvel[:, -1] - self.linvel[:, -2]) / self.dt
        self.angacc = (self.angvel[:, -1] - self.angvel[:, -2]) / self.dt

        self.t = torch.roll(self.t, shifts=-1, dims=1)
        self.t[:, -1] = self.t[:, -2] + self.dt

    def get_current_state(self):
        return {
            "position": self.position_history[:, -1],
            "quaternion_wxyz": self.quat_history[:, -1],
            "linvel": self.linvel[:, -1],
            "angvel": self.angvel[:, -1],
            "linacc": self.linacc,
            "angacc": self.angacc,
            "dt": self.dt
        }