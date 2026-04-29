import sumo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from sumo_rl import SumoEnvironment
import os
import subprocess
import gymnasium as gym
import numpy as np

def custom_reward_function(traffic_signal):
    """
    A robust reward function utilizing only native, verified sumo-rl methods
    from the TrafficSignal class source code.
    """
    # 1. Penalize total queue (Verified method line 287)
    queue_penalty = -traffic_signal.get_total_queued() 

    # 2. Penalize wait time utilizing the built-in diff-waiting-time logic (Verified method line 224)
    # This prevents the agent from hiding vehicles inside the intersection
    wait_time_diff = traffic_signal._diff_waiting_time_reward() 
    
    # We penalize both the queue AND the wait time diff
    reward = queue_penalty + wait_time_diff
    
    # Clip the reward to prevent massive updates during gridlock
    return max(-500, min(reward, 50))

class OcclusionWrapper(gym.ObservationWrapper):
    """
    Simulates sensor failure or camera occlusion by adding noise to observations.
    """
    def __init__(self, env, noise_level=0.15):
        super().__init__(env)
        self.noise_level = noise_level

    def observation(self, obs):
        noisy_obs = np.array(obs, dtype=np.float32) 
        noise = np.random.normal(loc=0.0, scale=self.noise_level * (np.abs(noisy_obs) + 1))
        noisy_obs = np.maximum(0.0, noisy_obs + noise)
        return noisy_obs

class RouteRegenerationCallback(BaseCallback):
    """
    Generates new traffic routes every N steps, smoothly scaling difficulty.
    """
    def __init__(self, regen_freq=25000, max_timesteps=300000, max_multiplier=2.0, verbose=0):
        super(RouteRegenerationCallback, self).__init__(verbose)
        self.regen_freq = regen_freq
        self.max_timesteps = max_timesteps
        self.max_multiplier = max_multiplier

    def _on_step(self) -> bool:
        if self.num_timesteps % self.regen_freq == 0:
            
            # Linear interpolation formula for smooth curriculum scaling
            # progress goes from 0.0 to 1.0 over the course of max_timesteps
            progress = min(1.0, self.num_timesteps / self.max_timesteps)
            
            # Calculate current multiplier (starts at 1.0, ends at max_multiplier)
            current_multiplier = 1.0 + (progress * (self.max_multiplier - 1.0))
            
            # Round to 2 decimal places to keep the CLI clean
            self.difficulty_multiplier = round(current_multiplier, 2)

            print(f"[{self.num_timesteps}] Regenerating routes... Difficulty scaled to: {self.difficulty_multiplier}x")
            
            try:
                ROOT = os.path.dirname(os.path.abspath(__file__))
                script = os.path.join(ROOT, "generate_routes.py")

                subprocess.run(
                    ["uv", "run", "python", script,
                    "--multiplier", str(self.difficulty_multiplier)],
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print(f"CRITICAL: Failed to generate routes: {e}")
                
        return True

def main():
    # 1. Initialize the Base Environment
    base_env = SumoEnvironment(
        net_file='net.net.xml',
        route_file='route.rou.xml',
        out_csv_name='outputs/ppo_robust_training_log',
        single_agent=True,
        use_gui=False,
        num_seconds=3600,
        reward_fn=custom_reward_function, # <--- Verified Reward Function
        delta_time=5,
        yellow_time=3,

        time_to_teleport=-1,  # Forces vehicles to stay in traffic jams forever
        sumo_warnings=False   # Hides the "tau lower than simulation step" console spam
    )

    # 2. Apply the Occlusion Wrapper
    noisy_env = OcclusionWrapper(base_env, noise_level=0.15) 

    # This is to know the reward and loss during training, and to ensure the environment is working correctly
    monitored_env = Monitor(noisy_env)

    # 3. Vectorize for SB3
    sb3_env = DummyVecEnv([lambda: monitored_env])

    # 4. Initialize PPO
    print("Starting Robust PPO Training...")
    model = PPO(
        "MlpPolicy", 
        sb3_env, 
        verbose=1, 
        learning_rate=1e-4, 
        batch_size=256,
        clip_range=0.1      
    )

    # Logging added
    log_path = "./outputs/sb3_logs/"
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # 5. Train with Route Regeneration Callback
    regen_callback = RouteRegenerationCallback(regen_freq=15000)
    model.learn(total_timesteps=500000, callback=regen_callback)

    # 6. Save
    model.save("ppo_robust_traffic_model")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()