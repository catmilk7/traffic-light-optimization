import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results():
    # 1. Load the data using absolute pathing to be safe
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(ROOT_DIR, "outputs", "sb3_logs", "progress.csv")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Could not find log file at {csv_path}")
        return

    # 2. Filter out rows where the reward hasn't been logged yet 
    # (like that very first iteration)
    if 'rollout/ep_rew_mean' not in df.columns:
        print("Reward data not found yet. The Monitor might not have finished an episode.")
        return
        
    df_reward = df.dropna(subset=['rollout/ep_rew_mean'])
    df_entropy = df.dropna(subset=['train/entropy_loss'])

    # 3. Set up the plotting grid (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Top Graph: Reward (Are we clearing traffic?) ---
    ax1.plot(df_reward['time/total_timesteps'], df_reward['rollout/ep_rew_mean'], 
             color='tab:green', linewidth=2, label='Average Episode Reward')
    ax1.set_title('Agent Performance (Reward)')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax1.legend()

    # --- Bottom Graph: Entropy (Are we exploring or stuck?) ---
    ax2.plot(df_entropy['time/total_timesteps'], df_entropy['train/entropy_loss'], 
             color='tab:orange', linewidth=2, label='Entropy Loss')
    ax2.set_title('Agent Exploration (Entropy)')
    ax2.set_xlabel('Total Timesteps')
    ax2.set_ylabel('Entropy')
    ax2.grid(True)
    ax2.legend()

    # 4. Render
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_results()