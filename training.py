import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback  # NEW: For auto-saving
from sumo_rl import SumoEnvironment


def custom_reward_function(traffic_signal):
    queue_penalty = -traffic_signal.get_total_queued()
    wait_time_diff = traffic_signal._diff_waiting_time_reward()
    pressure = traffic_signal.get_pressure()
    pressure_penalty = -0.5 * abs(pressure)
    reward = queue_penalty + wait_time_diff + pressure_penalty
    return max(-500, min(reward, 50))


class OcclusionWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_level=0.15):
        super().__init__(env)
        self.noise_level = noise_level

    def observation(self, obs):
        noisy_obs = np.array(obs, dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=self.noise_level * (np.abs(noisy_obs) + 1))
        noisy_obs = np.maximum(0.0, noisy_obs + noise)
        return noisy_obs


class ScalableEnv(SumoEnvironment):
    SCALE_CHOICES = [0.5, 1.0, 1.5, 2.0]

    def reset(self, **kwargs):
        scale = float(np.random.choice(self.SCALE_CHOICES))
        self.additional_sumo_cmd = f"--scale {scale}"
        print(f"[Episode reset] Traffic scale: {scale}x")
        return super().reset(**kwargs)


def main():
    # Ensure output directories exist
    os.makedirs("./outputs/checkpoints", exist_ok=True)
    os.makedirs("./outputs/sb3_logs", exist_ok=True)

    # 1. Update factory to accept teleport_time dynamically
    def make_env(teleport_time, phase_name):
        base_env = ScalableEnv(
            net_file='net.net.xml',
            route_file='route.rou.xml',
            out_csv_name=f'outputs/ppo_traffic_log_{phase_name}', # Separate CSVs per phase
            single_agent=True,
            use_gui=False,
            num_seconds=7200,
            reward_fn=custom_reward_function,
            delta_time=5,
            yellow_time=3,
            time_to_teleport=teleport_time,   # PARAMETERIZED
            sumo_warnings=False    
        )
        noisy_env = OcclusionWrapper(base_env, noise_level=0.15)
        return Monitor(noisy_env)

    # 2. Setup the Checkpoint Callback (Saves every 50k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./outputs/checkpoints/",
        name_prefix="robust_ppo"
    )

    # ==========================================
    # PHASE 1: THE TRAINING WHEELS (250k steps)
    # ==========================================
    print("=== STARTING PHASE 1: Training Wheels (teleport=300) ===")
    env_phase1 = DummyVecEnv([lambda: make_env(300, "phase1")])
    
    model = PPO(
        "MlpPolicy",
        env_phase1,
        verbose=1,
        learning_rate=1e-4,
        n_steps=1440,      
        batch_size=480,    
        clip_range=0.1,
        ent_coef=0.01      
    )

    new_logger = configure("./outputs/sb3_logs/", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=250000, callback=checkpoint_callback)
    model.save("./outputs/ppo_phase1_complete")

    # ==========================================
    # PHASE 2: HARDCORE MODE (250k steps)
    # ==========================================
    print("=== STARTING PHASE 2: Hardcore Mode (teleport=-1) ===")
    env_phase2 = DummyVecEnv([lambda: make_env(-1, "phase2")])
    
    # Swap the environment inside the ALREADY TRAINED model
    model.set_env(env_phase2)
    
    # Resume learning (reset_num_timesteps=False keeps the logger counting to 500k smoothly)
    model.learn(total_timesteps=250000, callback=checkpoint_callback, reset_num_timesteps=False)
    
    model.save("./outputs/ppo_final_hardcore_model")
    print("=== TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()