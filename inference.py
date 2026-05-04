import sumo
import gymnasium as gym
from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment

# --- 1. Configuration ---
MODEL_PATH = r"outputs\ppo_final_hardcore_model" # Points to my_ppo_model.zip (extension is optional in SB3)
NUM_EPISODES = 5            # How many simulation runs you want to observe

def custom_reward_function(traffic_signal):
    # 1. Queue Penalty
    # get_total_queued() returns a positive integer. We must manually make it negative.
    queue_penalty = -0.1 * traffic_signal.get_total_queued()

    # 2. Wait Time Diff
    # The source code handles this perfectly. If wait time drops, it returns positive (good).
    wait_time_diff = traffic_signal._diff_waiting_time_reward()

    # 3. Pressure Reward (THE BIG FIX)
    # Because sumo-rl defines pressure as (Out - In), positive pressure means the intersection is clearing.
    # We ADD it now, instead of subtracting it.
    pressure_reward = 0.1 * traffic_signal.get_pressure()

    # 4. Living Reward (Baseline)
    # Keeps the agent optimistic. Doing nothing while traffic flows equals a positive score.
    baseline = 1.5 

    reward = baseline + queue_penalty + wait_time_diff + pressure_reward

    # 5. Symmetric Clipping
    return max(-100, min(reward, 100))

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
        reward_fn=custom_reward_function,
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