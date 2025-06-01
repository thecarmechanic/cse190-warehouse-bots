import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("episode_stats.csv")  # replace with your actual filename

# Create the plot
fig, ax1 = plt.subplots()

# Plot total_reward on the primary y-axis
color = 'tab:blue'
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Reward', color=color)
ax1.plot(df['episode'], df['total_reward'], label='Total Reward', color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for avg_loss
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Avg Loss', color=color)
ax2.plot(df['episode'], df['avg_loss'], label='Avg Loss', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Add title and show
plt.title('Episode vs Total Reward and Avg Loss')
fig.tight_layout()
plt.show()
