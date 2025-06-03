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

# Your layout and env setup
layout = """
..g..
.x.x.
.x.x.
.x.x.
"""

medium_layout = """
..g...g.....
x....x.....x
x..........x
x...........
.x.x.x.x.x.x
"""

n_agents = 2
n_actions = 5
max_steps = 100
height = 3
width = 7

env = gym.make("rware:rware-tiny-2ag-v2", layout=medium_layout)
obs, info = env.reset()
print(info)

# --- get direction ---
def get_direction(agent_obs):
    direction_one_hot = agent_obs[3:7]
    direction_index = direction_one_hot.argmax()
    return direction_index

# --- will any agents crash? ---
# TODO

# --- can an agent go forward given an env and its direction ---
def can_go_forward(env, x, y, direction, carrying):
    go_forward = [x, y]
    if direction == 0:
        go_forward[1] -= 1
    elif direction == 1:
        go_forward[1] += 1
    elif direction == 2:
        go_forward[0] -= 1
    else: 
        go_forward[0] += 1

    print(f"go forward: {go_forward}")

    if go_forward[0] < 0 or go_forward[0] >= width:
        print("width is bad??")
        return False
    if go_forward[1] < 0 or go_forward[1] >= height:
        print("height is bad??")
        return False

    if carrying:
        print(env.unwrapped.grid)
        print("oooo", env.unwrapped.grid[1])
        if env.unwrapped.grid[1][int(go_forward[1])][int(go_forward[0])]:
            print("there's a shelf in front???")
            return False
        
    print(x, y, go_forward, direction)
    
    
    
    return True


# --- valid actions method --- (TODO)
def get_valid_actions(env, agent_obs):
    return [1, 2, 3, 4]

# Assuming obs_dim is the flattened size of one agent's observation
obs_dim = np.array(obs[0]).flatten().shape[0]

# Define your DQN class here (or import it)
class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        )
    def forward(self, x):
        return self.net(x)

# Initialize and load weights
shared_policy_net = DQN(obs_dim, n_actions)
shared_policy_net.load_state_dict(torch.load('shared_dqn_model.pth'))
shared_policy_net.eval()

def select_greedy_action(policy_net, obs, valid_actions):
    flat_obs = np.array(obs).flatten()
    obs_tensor = torch.FloatTensor(flat_obs).unsqueeze(0)
    with torch.no_grad():
        q_values = policy_net(obs_tensor).squeeze()
    q_values_filtered = torch.full_like(q_values, float('-inf'))
    for a in valid_actions:
        q_values_filtered[a] = q_values[a]
    return int(q_values_filtered.argmax())

done = [False] * n_agents
frames = []

total_reward = 0
for step in range(max_steps):
    actions = []
    for i in range(n_agents):
        valid_actions = get_valid_actions(env, obs[i])
        action = select_greedy_action(shared_policy_net, obs[i], valid_actions)
        actions.append(action)
        print(f"Agent {i} action: {action}")
    
    obs, rewards, terminated, truncated, info = env.step(actions)
    print("Rewards:", rewards)
    for reward in rewards:
        total_reward += reward

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

imageio.mimsave("rware_dqn_eval_load_and_run.gif", frames, fps=10)
print("🎥 Saved rware_dqn_eval.gif")
print(f"Total rewards: {total_reward}")

input("🏁 Press Enter to exit...")
env.close()
