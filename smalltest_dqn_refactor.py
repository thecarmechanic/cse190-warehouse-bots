import gymnasium as gym
import rware
import numpy as np
import time
import imageio
from pyglet import image
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import csv

# --- configs ---
n_agents = 2
n_actions = 5
alpha = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.05
num_episodes = 3000
max_steps = 75
batch_size = 128
target_update_freq = 10
replay_capacity = 10000
max_reward = 50
progress_check = 100    # how often we want to print episode progress in training loop

frames = []

# --- layout ---
layout = """
..x..
.g.g.
.x..x
"""

# --- environment ---
env = gym.make("rware:rware-tiny-2ag-v2", layout=layout)
obs, info = env.reset()
print(info)

# --- flatten observation to represent state --- (???)
def flatten_obs(obs_tuple):
    flattened = np.concatenate([np.array(part).flatten() for part in obs_tuple])
    print(flattened)
    return flattened

# get the dimension of a flattened observation (for one agent)
obs_dim = np.array(obs[0]).flatten().shape[0]

# --- neural net ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(), 
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
    
# --- initialized the shared components (target net, policy net, optimizer, memory) ---
shared_policy_net = DQN(obs_dim, n_actions).to(device)
shared_target_net = DQN(obs_dim, n_actions).to(device)
shared_target_net.load_state_dict(shared_policy_net.state_dict())
shared_optimizer = optim.Adam(shared_policy_net.parameters(), lr=alpha)
shared_memory = deque(maxlen=replay_capacity)

# --- agent class ---
class Agent:
    def __init__(self, id, policy_net, target_net, optimizer, memory):
        self.id = id
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.memory = memory
    
    def select_action(self, obs, eps):
        if random.random() < eps:
            actions = env.action_space[self.id]
            chosen_action = actions.sample()
            # print(actions, chosen_action)
            return chosen_action
        obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor)
        return q_values.argmax().item()
    
    def store(self, s, a, r, s_next, done):
        self.memory.append((np.array(s).flatten(), a, r, np.array(s_next).flatten(), done))
    
    def train_step(self):
        if len(self.memory) < batch_size:
            return None
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=device)

        q_values = self.policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            max_net_q = self.target_net(next_states).max(1)[0]
            targets = rewards + gamma * max_net_q * (1 - dones)
        
        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# --- initialize agents ---
agents = [Agent(i, shared_policy_net, shared_target_net, shared_optimizer, shared_memory) for i in range(n_agents)]

episode_stats = []  # before the loop

# --- training loop ---
for episode in range(num_episodes):
    episode_losses = []
    obs, _ = env.reset()
    total_reward = [0] * n_agents
    done = [False] * n_agents
    visited_states = set()
    
    for i in range(n_agents):
        visited_states.add((i, tuple(obs[i].flatten())))
    
    agent_steps = [max_steps] * n_agents # track the steps each agent takes
    for step in range(max_steps):
        actions = [agent.select_action(obs[i], epsilon) for i, agent in enumerate(agents)]
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        # print(info)

        terminated = [terminated] * n_agents if isinstance(terminated, bool) else terminated
        truncated = [truncated] * n_agents if isinstance(truncated, bool) else truncated
        done = [t or tr for t, tr in zip(terminated, truncated)]

        for i in range(n_agents):
            agents[i].store(obs[i], actions[i], rewards[i], next_obs[i], done[i])
            loss = agents[i].train_step()
            if loss is not None:
                episode_losses.append(loss)
            total_reward[i] += rewards[i]
            
            if done[i]:
                agent_steps[i] = step
        
        obs = next_obs
        complete_reward = 0
        for r in total_reward:
            complete_reward += r

        if all(done) or complete_reward >= max_reward:
            break
    
    avg_loss = np.mean(episode_losses) if episode_losses else 0.0
    total_reward_sum = sum(total_reward)
    avg_reward_per_agent = total_reward_sum / n_agents
    
    
    # --- epsilon decay ---
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # --- update target net? ---
    if (episode + 1) % target_update_freq == 0:
        shared_target_net.load_state_dict(shared_policy_net.state_dict())
    
    # --- print progress? ---
    if (episode + 1) % progress_check == 0:
        print(f"Episode {episode+1}, epsilon={epsilon:.3f}, "
              f"total_reward={total_reward_sum}, avg_reward={avg_reward_per_agent:.2f}, "
              f"avg_loss={avg_loss:.4f}")
    
    episode_stats.append({
        "episode": episode + 1,
        "epsilon": epsilon,
        "total_reward": total_reward_sum,
        "avg_reward": avg_reward_per_agent,
        "avg_loss": avg_loss,
        "steps": agent_steps
    })

print("\n✅ DQN Training Complete!")

torch.save(shared_policy_net.state_dict(), "shared_dqn_model.pth")
print("✅ Model saved to 'shared_dqn_model.pth'")

with open("episode_stats.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=episode_stats[0].keys())
    writer.writeheader()
    writer.writerows(episode_stats)

# --- evaluation ---
obs, _ = env.reset()
done = [False] * n_agents

for step in range(max_steps):
    actions = []
    for i in range(n_agents):
        flat_obs = np.array(obs[i]).flatten()
        obs_tensor = torch.tensor(np.array(flat_obs), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_vals = shared_policy_net(obs_tensor).detach().cpu().numpy()
        action = int(q_vals.argmax())
        actions.append(action)
        print(f"Agent {i} action: {action}")

    obs, rewards, terminated, truncated, infos = env.step(actions)
    # print(info, "hello?")
    terminated = [terminated] * n_agents if isinstance(terminated, bool) else terminated
    truncated = [truncated] * n_agents if isinstance(truncated, bool) else truncated
    done = [t or tr for t, tr in zip(terminated, truncated)]

    env.render()
    if hasattr(env.unwrapped, "renderer") and hasattr(env.unwrapped.renderer, "window"):
        win = env.unwrapped.renderer.window
        win.switch_to()
        win.dispatch_events()
        buffer = image.get_buffer_manager().get_color_buffer()
        img_data = buffer.get_image_data()
        frame = np.frombuffer(img_data.get_data('RGB', buffer.width * 3), dtype=np.uint8)
        frame = frame.reshape((buffer.height, buffer.width, 3))
        frame = np.flipud(frame)
        frames.append(frame)

    time.sleep(0.1)

    if all(done):
        break

# --- save GIF ---
imageio.mimsave("rware_dqn_eval.gif", frames, fps=10)
print("🎥 Saved rware_dqn_eval.gif")

input("🏁 Press Enter to exit...")
env.close()