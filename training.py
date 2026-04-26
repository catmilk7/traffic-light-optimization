import sumo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from sumo_rl import SumoEnvironment
import os
import subprocess
import gymnasium as gym
import numpy as np

def custom_reward_function(traffic_signal):
    """
    A balanced reward function to prevent cheating and encourage throughput.
    """
    # 1. Heavily penalize the total number of stopped vehicles (queue length)
    queue_penalty = -traffic_signal.get_total_queued() 

    # 2. Penalize total accumulated waiting time 
    # FIX: Fetch the list of waiting times per lane, then sum them up
    lane_waiting_times = traffic_signal.get_accumulated_waiting_time_per_lane()
    wait_time = sum(lane_waiting_times) 
    wait_penalty = -(wait_time * 0.01) 

    # 3. Penalize phase changes (prevents rapid flickering to trap cars)
    # Dynamically create an attribute to track the previous phase
    if not hasattr(traffic_signal, "custom_last_phase"):
        traffic_signal.custom_last_phase = traffic_signal.phase

    # If the light changed this step, apply a penalty
    phase_change_penalty = -5 if traffic_signal.custom_last_phase != traffic_signal.phase else 0
    traffic_signal.custom_last_phase = traffic_signal.phase # Store for next step

    # Sum the components
    reward = queue_penalty + wait_penalty + phase_change_penalty
    
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
        # obs is typically an array of queue lengths, wait times, etc.
        # We apply Gaussian noise proportional to the current value
        
        # Ensure we don't try to add noise to discrete data if it exists in your obs space
        noisy_obs = np.array(obs, dtype=np.float32) 
        
        # Generate noise
        noise = np.random.normal(loc=0.0, scale=self.noise_level * (np.abs(noisy_obs) + 1))
        
        # Apply noise, ensuring values don't drop below 0 for queues/times
        noisy_obs = np.maximum(0.0, noisy_obs + noise)
        
        return noisy_obs

class RouteRegenerationCallback(BaseCallback):
    """
    A custom callback that generates new traffic routes every N steps.
    It implements Curriculum Learning by increasing traffic density over time.
    """
    def __init__(self, regen_freq=15000, verbose=0):
        super(RouteRegenerationCallback, self).__init__(verbose)
        self.regen_freq = regen_freq
        self.difficulty_multiplier = 1.0

    def _on_step(self) -> bool:
        # Trigger route regeneration every `regen_freq` steps
        if self.num_timesteps % self.regen_freq == 0:
            
            # Curriculum Logic: Increase traffic based on training progress
            if self.num_timesteps >= 200000:
                self.difficulty_multiplier = 3.0  # 3x traffic (Rush Hour)
            elif self.num_timesteps >= 100000:
                self.difficulty_multiplier = 2.0  # 2x traffic (Heavy)
            else:
                self.difficulty_multiplier = 1.0  # 1x traffic (Base)

            print(f"[{self.num_timesteps}] Regenerating routes... Difficulty: {self.difficulty_multiplier}x")

            try:
                # Call the external script, passing the multiplier
                subprocess.run(
                    ["uv", "run", "generate_routes.py", "--multiplier", str(self.difficulty_multiplier)],
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
        reward_fn=custom_reward_function, # <--- Use the custom anti-cheat reward
        delta_time=5,
        yellow_time=3 
    )

    # 2. Apply the Occlusion Wrapper
    noisy_env = OcclusionWrapper(base_env, noise_level=0.15) # 15% sensor uncertainty

    # 3. Vectorize for SB3
    sb3_env = DummyVecEnv([lambda: noisy_env])

    # 4. Initialize PPO with a slightly lower learning rate for stability
    print("Starting Robust PPO Training...")
    model = PPO(
        "MlpPolicy", 
        sb3_env, 
        verbose=1, 
        learning_rate=1e-4, # Lowered from 3e-4 to handle noisy environments better
        batch_size=256,
        clip_range=0.1      # Tighter clip to prevent massive policy shifts during gridlock
    )

    # 5. Train with Route Regeneration Callback
    regen_callback = RouteRegenerationCallback(regen_freq=15000)
    
    # Train for a longer duration since the environment is harder now
    model.learn(total_timesteps=500000, callback=regen_callback)

    # 6. Save
    model.save("ppo_robust_traffic_model")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()