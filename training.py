import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback  # NEW: For auto-saving
from sumo_rl import SumoEnvironment


import numpy as np
from sumo_rl import SumoEnvironment

def stable_traffic_reward(trafic_signal):
    """
    Standard SumoEnvironment calls this with the 'TrafficSignal' instance.
    No subclassing required.
    """
    # 1. Pressure: (Out - In). Positive is good.
    # High pressure means we are successfully moving cars out of the bottleneck.
    pressure = trafic_signal.get_pressure()
    pressure_reward = 0.2 * pressure if pressure > 0 else 0.1 * pressure

    # 2. Max Queue Penalty: The "Anti-Starvation" anchor.
    # Instead of squaring total time, we penalize the most congested lane.
    # This prevents side-street starvation without mathematical instability.
    queues = trafic_signal.get_lanes_queue()  # Returns list of density [0, 1] per lane
    max_queue_penalty = -2.0 * max(queues) if queues else 0

    # 3. Wait Time Delta: Short-term trend.
    wait_diff = trafic_signal._diff_waiting_time_reward()

    # 4. Total Halting Penalty: General congestion check.
    total_halting = -0.1 * trafic_signal.get_total_queued()

    # 5. Living Reward
    baseline = 1.0

    return baseline + pressure_reward + max_queue_penalty + wait_diff + total_halting

# --- Environment Setup ---
def make_env(teleport_time):
    # Use the base SumoEnvironment and pass the function directly
    env = SumoEnvironment(
        net_file='net.net.xml',
        route_file='route.rou.xml',
        single_agent=True,
        use_gui=False,
        num_seconds=7200,
        reward_fn=stable_traffic_reward, # Pass function here
        delta_time=5,
        yellow_time=3,
        time_to_teleport=teleport_time,
        sumo_warnings=False
    )
    return env


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
            reward_fn=stable_traffic_reward,
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
        learning_rate=3e-5,
        n_steps=1440,      
        batch_size=480,    
        clip_range=0.1,
        ent_coef=0.05      
    )

    new_logger = configure("./outputs/sb3_logs/", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=250000, callback=checkpoint_callback)
    model.save("./outputs/ppo_phase1_complete")

    # ==========================================
    # PHASE 2: HARDCORE MODE (250k steps)
    # ==========================================
    print("=== STARTING PHASE 2: The next normal Mode (teleport=300) ===")
    env_phase2 = DummyVecEnv([lambda: make_env(300, "phase2")])
    
    # Swap the environment inside the ALREADY TRAINED model
    model.set_env(env_phase2)
    
    # Resume learning (reset_num_timesteps=False keeps the logger counting to 500k smoothly)
    model.learn(total_timesteps=250000, callback=checkpoint_callback, reset_num_timesteps=False)
    
    model.save("./outputs/ppo_final_hardcore_model")
    print("=== TRAINING COMPLETE ===")


if __name__ == "__main__":
    main()