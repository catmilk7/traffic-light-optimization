import sumo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sumo_rl import SumoEnvironment

def main():
    # 1. THE BRIDGE: Connect to your friend's SUMO files
    env = SumoEnvironment(
        net_file='net.net.xml',
        route_file='route.rou.xml',
        out_csv_name='outputs/ppo_training_log', # Updated to match PPO
        single_agent=True,   # <--- This makes it a standard Gym Env!
        use_gui=False,       # Headless for speed
        num_seconds=3600,    # 1 episode = 1 hour of traffic
        reward_fn='diff-waiting-time' 
    )

    # 2. THE WRAPPER: Native SB3
    # We DELETE supersuit entirely. We just wrap the native Gym 
    # environment directly into SB3's DummyVecEnv.
    sb3_env = DummyVecEnv([lambda: env])

    # 3. THE BRAIN: Initialize PPO
    print("Starting PPO Training...")
    model = PPO(
        "MlpPolicy", 
        sb3_env, 
        verbose=1, 
        learning_rate=3e-4,
        batch_size=256
    )

    # Train for your target steps
    model.learn(total_timesteps=300000)

    # 4. THE EXPORT: Save the model
    model.save("ppo_traffic_model")
    print("Training complete. Model saved as ppo_traffic_model.zip")

if __name__ == "__main__":
    main()