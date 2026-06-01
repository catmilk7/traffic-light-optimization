import sumo
import gymnasium as gym
from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment

# --- 1. Configuration ---
MODEL_PATH = r"outputs\ppo_final_hardcore_model" # Points to my_ppo_model.zip (extension is optional in SB3)
NUM_EPISODES = 5            # How many simulation runs you want to observe

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

def run_inference():
    # --- 2. Load the Environment ---
    # Replace this with exactly how you initialized your SUMO environment for training.
    # Make sure to set GUI=True if you want to visually watch the traffic lights!
    env = SumoEnvironment(
        net_file='net.net.xml',
        route_file='route.rou.xml',
        single_agent=True,
        use_gui=True,
        num_seconds=7200,
        reward_fn=stable_traffic_reward,
        delta_time=5,
        yellow_time=3,
        time_to_teleport=300,   # PARAMETERIZED
        sumo_warnings=False
    )
    
    # --- 3. Load the Trained Model ---
    print(f"Loading trained model from {MODEL_PATH}.zip...")
    model = PPO.load(MODEL_PATH, env=env)
    
    # --- 4. Evaluation Loop ---
    for episode in range(NUM_EPISODES):
        # Reset environment for a new episode
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        print(f"--- Starting Episode {episode + 1} ---")
        
        while not (done or truncated):
            # The magic happens here: deterministic=True turns off exploration
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
        print(f"Episode {episode + 1} finished in {step_count} steps.")
        print(f"Total Cumulative Reward: {episode_reward:.2f}\n")
        
    # Clean up
    env.close()
    print("Inference complete.")

if __name__ == "__main__":
    run_inference()