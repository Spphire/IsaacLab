"""
Shadow Hand Ur10e Arm environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

handarm_task_entry = "isaaclab_tasks.direct.handarm_manipulation"

gym.register(
    id="shadow-obs163",
    entry_point=f"{handarm_task_entry}.handarm_manipulation_env:HandArmManipulationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.shadow_hand_ur10e_arm_env_cfg:ShadowHandUr10eArmEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)