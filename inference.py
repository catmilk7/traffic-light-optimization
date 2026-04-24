import sumo
import gymnasium as gym
from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment

# --- 1. Configuration ---
MODEL_PATH = r"C:\Users\ADMIN\Documents\GitHub\traffic-light-optimization\ppo_traffic_model.zip" # Points to my_ppo_model.zip (extension is optional in SB3)
NUM_EPISODES = 5            # How many simulation runs you want to observe

def run_inference():
    # --- 2. Load the Environment ---
    # Replace this with exactly how you initialized your SUMO environment for training.
    # Make sure to set GUI=True if you want to visually watch the traffic lights!
    env = SumoEnvironment(
        net_file='net.net.xml',
        route_file='route.rou.xml',
        out_csv_name='outputs/ppo_inference_log',
        single_agent=True,   # <--- This makes it a standard Gym Env!
        use_gui=True,       # Headless for speed
        num_seconds=3600,    # 1 episode = 1 hour of traffic
        reward_fn='diff-waiting-time',
        
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