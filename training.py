import supersuit as ss
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from sumo_rl import SumoEnvironment

def main():
    # 1. THE BRIDGE: Connect to your friend's SUMO files
    # Note: Because you installed libsumo, sumo-rl will automatically use it!
    env = SumoEnvironment(
        net_file='net.net.xml',
        route_file='route.rou.xml',
        out_csv_name='outputs/sac_training_log',
        single_agent=True,
        use_gui=False,       # Headless for speed
        num_seconds=3600,    # 1 episode = 1 hour of traffic
        reward_fn='diff-waiting-time' # The built-in delta reward
    )

    # 2. THE WRAPPER: Format the environment for SB3
    # sumo-rl uses PettingZoo natively, so we convert it to an SB3 Vector Env
    sb3_env = ss.pettingzoo_env_to_vec_env_v1(env)
    sb3_env = ss.concat_vec_envs_v1(sb3_env, 1, num_cpus=1, base_class='stable_baselines3')

    # 3. THE BRAIN: Initialize SAC
    print("Starting SAC Training...")
    model = SAC(
        "MlpPolicy", 
        sb3_env, 
        verbose=1, 
        learning_rate=0.0003,
        batch_size=256
    )

    # Train for your target steps
    model.learn(total_timesteps=300000)

    # 4. THE EXPORT: Save the model to hand off to the hardware team
    model.save("sac_traffic_model")
    print("Training complete. Model saved as sac_traffic_model.zip")

if __name__ == "__main__":
    main()