import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


SHADOW_HAND_UR10E_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="assets/shadowhand_ur10e.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        #collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
                    #"rh.*J.*": 0.0,
                    "rh_WRJ1": 0.0,
                    "rh_WRJ2": 0.0,
                    "rh_THJ3": 0.0,
                    "rh_THJ5":-1.046,#-1.6, #rou
                    "rh_THJ4":0.3, #yaw
                    "rh_THJ2":0.690,#1.4, #p
                    "rh_THJ1":0.4,
                    "rh_LFJ(5|4)":0.0,
                    "rh_LFJ(3|2|1)":1.4,
                    "rh_MFJ4":0.0,
                    "rh_MFJ(3|2|1)":0.8,
                    "rh_FFJ4":0.0,
                    "rh_FFJ(3|2|1)":0.8,
                    "rh_RFJ4":0.0,
                    "rh_RFJ(3|2|1)":0.8,
                   "ra_shoulder_pan_joint": 0.0004430418193805963,
            "ra_shoulder_lift_joint": -1.6971481482135218, #-1.712,
            "ra_elbow_joint": 2.4696645736694336, #1.712,
            "ra_wrist_1_joint": -0.6649912039386194,
            "ra_wrist_2_joint": 1.6591527462005615,
            "ra_wrist_3_joint": 0.014633487910032272,},
        joint_vel={"rh.*J.*": 0.0,
                   "ra_shoulder_pan_joint": 0.0,
            "ra_shoulder_lift_joint": 0.0,
            "ra_elbow_joint": 0.0,
            "ra_wrist_1_joint": 0.0,
            "ra_wrist_2_joint": 0.0,
            "ra_wrist_3_joint": 0.0,},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["rh_WR.*", "rh_(FF|MF|RF|LF|TH)J(4|3|2|1)", "rh_(LF|TH)J5", "ra.*joint"],
            effort_limit={
                "rh_WRJ2": 4.785,
                "rh_WRJ1": 2.175,
                "rh_(FF|MF|RF|LF)J2": 0.7245,
                "rh_(FF|MF|RF|LF)J1": 0.81,
                "rh_FFJ(4|3)": 0.9,
                "rh_MFJ(4|3)": 0.9,
                "rh_RFJ(4|3)": 0.9,
                "rh_LFJ(5|4|3)": 0.9,
                "rh_THJ5": 2.3722,
                "rh_THJ4": 1.45,
                "rh_THJ(3|2)": 0.99,
                "rh_THJ1": 0.81,
                "ra_.*_joint": 87.0,
            },
            stiffness={
                "rh_WRJ.*": 5.0,
                "rh_(FF|MF|RF|LF|TH)J(4|3|2|1)": 1.0,
                "rh_(LF|TH)J5": 1.0,
                ".*_joint": 800.0,
            },
            damping={
                "rh_WRJ.*": 0.5,
                "rh_(FF|MF|RF|LF|TH)J(4|3|2|1)": 0.1,
                "rh_(LF|TH)J5": 0.1,
                "ra_.*_joint": 40.0,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

@configclass
class ShadowHandUr10eArmEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_space = 30
    observation_space = 163
    state_space = 0
    asymmetric_obs = False

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=600000,
        ),
    )
    # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_UR10E_ARM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    actuated_joint_names = [
        "ra_shoulder_pan_joint",
        "ra_shoulder_lift_joint",
        "ra_elbow_joint",
        "ra_wrist_1_joint",
        "ra_wrist_2_joint",
        "ra_wrist_3_joint",
        "rh_WRJ2",
        "rh_WRJ1",
        "rh_FFJ4",
        "rh_FFJ3",
        "rh_FFJ2",
        "rh_FFJ1",

        "rh_MFJ4",
        "rh_MFJ3",
        "rh_MFJ2",
        "rh_MFJ1",

        "rh_RFJ4",
        "rh_RFJ3",
        "rh_RFJ2",
        "rh_RFJ1",

        "rh_LFJ5",
        "rh_LFJ4",
        "rh_LFJ3",
        "rh_LFJ2",
        "rh_LFJ1",

        "rh_THJ5",
        "rh_THJ4",
        "rh_THJ3",
        "rh_THJ2",
        "rh_THJ1",


    ]
    fingertip_body_names = [
        "rh_ffdistal",
        "rh_mfdistal",
        "rh_rfdistal",
        "rh_lfdistal",
        "rh_thdistal",
    ]

    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
            scale=(1, 1, 1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.76, 0.12, 0.26), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1, 1, 1),
            )
        },
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=5, replicate_physics=True)

    # parameters
    keypoints_scale = 0.2
    forearm_rot_target = [0.5, -0.5,  0.5, -0.5]
    fall_threshold = 0.04

    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    forearm_linvel_error_threshold = 0.2
    forearm_rotvel_error_threshold = 0.4
    object_acc_diff_threshold = 0.1
    object_acc_penalty = 20
    
    forearm_linvel_error_weight = 0.25
    forearm_rotvel_error_weight = 0.25
    forearm_rot_error_weight = 0.5
    pos_error_weight = 2.5
    
    total_reward_scale = 10
    total_reward_eps = 1e-6
    
    subepoch_threshold = 20000
    
    pos_success_threshold = 0.02
    forearm_rotvel_success_threshold = 0.3
    keypoints_success_threshold = 0.01
    
    reach_goal_bonus = 250
    fall_threshold = 0.04
    fall_penalty = -50.0
    av_factor = 0.1
    
