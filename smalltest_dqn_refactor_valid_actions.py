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
epsilon_decay = 0.9955
epsilon_min = 0.05
num_episodes = 4000
max_steps = 75
batch_size = 64
target_update_freq = 10
replay_capacity = 10000
max_reward = 50
progress_check = 100    # how often we want to print episode progress in training loop

frames = []

# --- layout ---
layout = """
..g..
.x.x.
.x.x.
.x.x.
"""
width = 5
height = 4

# --- environment ---
env = gym.make("rware:rware-tiny-2ag-v2", layout=layout)
obs, info = env.reset()
print(info)

# --- flatten observation to represent state --- (???)
def flatten_obs(obs_tuple):
    flattened = np.concatenate([np.array(part).flatten() for part in obs_tuple])
    print(flattened)
    return flattened

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

    # print(f"go forward: {go_forward}")

    if go_forward[0] < 0 or go_forward[0] >= width:
        # print("width is bad??")
        return False
    if go_forward[1] < 0 or go_forward[1] >= height:
        # print("height is bad??")
        return False

    if carrying:
        #print(env.unwrapped.grid)
        #print("oooo", env.unwrapped.grid[1])
        if env.unwrapped.grid[1][int(go_forward[1])][int(go_forward[0])]:
            #print("there's a shelf in front???")
            return False
        if (go_forward[0], go_forward[1]) in env.unwrapped.goals:
            return True
        
    # print(x, y, go_forward, direction)
    
    return True

# --- valid actions method --- (TODO)
def get_valid_actions(env, agent_obs):
    valid_actions = []
    x = agent_obs[0]
    y = agent_obs[1]
    is_carrying = (agent_obs[2] != 0)
    direction = get_direction(agent_obs)
    i_can_go_forward = can_go_forward(env, x, y, direction, is_carrying)
    shelf_at_my_location = env.unwrapped.grid[1][int(y)][int(x)]
    goals = env.unwrapped.goals
    at_a_goal = any(goal[0] == y and goal[1] == x for goal in goals)

    can_toggle = False
    if is_carrying:
        if at_a_goal:
            can_toggle = True
    else:
        if shelf_at_my_location != 0:
            can_toggle = True

    if can_toggle:
        #print(f"{x}, {y}. CAN toggle: is_carrying: {is_carrying}, shelf_at_my_location: {shelf_at_my_location}, at_a_goal: {at_a_goal}, goals: {goals}")
        valid_actions.append(4)
    else:
        pass
        #print(f"{x}, {y}. can't toggle: is_carrying: {is_carrying}, shelf_at_my_location: {shelf_at_my_location}, at_a_goal: {at_a_goal}, goals: {goals}")
    if i_can_go_forward:
        valid_actions.append(1)
    
    valid_actions.append(0)
    valid_actions.append(2)
    valid_actions.append(3)

    return list(set(valid_actions))

# get the dimension of a flattened observation (for one agent)
obs_dim = np.array(obs[0]).flatten().shape[0]

# --- neural net ---c
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
    
# --- initialized the shared components (target net, policy net, optimizer, memory) ---
shared_policy_net = DQN(obs_dim, n_actions)
shared_target_net = DQN(obs_dim, n_actions)
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
    
    def select_action(self, obs, next_valid_actions, eps):
        #print(obs, "**")
        if random.random() < eps:
            return random.choice(next_valid_actions)
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor).squeeze()
        
        q_values_numpy = q_values.cpu().numpy()
        masked_q_values = np.full_like(q_values_numpy, -np.inf)
        for a in next_valid_actions:
            masked_q_values[a] = q_values_numpy[a]
    
        return int(np.argmax(masked_q_values))
    
    def store(self, s, a, r, s_next, done, valid_actions, next_valid_actions):
        self.memory.append((np.array(s).flatten(), a, r, np.array(s_next).flatten(), done, valid_actions, next_valid_actions))
    
    def train_step(self):
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, valid_actions_batch, next_valid_actions_batch = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        # Q-values for current state-action pairs
        q_values = self.policy_net(states).gather(1, actions).squeeze()

        # Compute target Q-values with only valid next actions
        with torch.no_grad():
            next_q_values_all = self.target_net(next_states)
            max_q_values = []

            for i in range(batch_size):
                valid = next_valid_actions_batch[i]
                if len(valid) > 0:
                    valid_qs = next_q_values_all[i][valid]
                    max_q = valid_qs.max().item()
                else:
                    max_q = 0.0  # No valid actions → no future value
                max_q_values.append(max_q)

            max_q_values = torch.FloatTensor(max_q_values)
            targets = rewards + gamma * max_q_values * (1 - dones)

        # Loss and optimizer step
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

    for step in range(max_steps):
        actions = []
        valid_actions_list = []

        for i in range(n_agents):
            valid_actions = get_valid_actions(env, obs[i])
            valid_actions_list.append(valid_actions)
            action = agents[i].select_action(obs[i], valid_actions, epsilon)
            actions.append(action)

        # Step environment
        next_obs, rewards, terminated, truncated, info = env.step(actions)

        terminated = [terminated] * n_agents if isinstance(terminated, bool) else terminated
        truncated = [truncated] * n_agents if isinstance(truncated, bool) else truncated
        done = [t or tr for t, tr in zip(terminated, truncated)]

        # ✅ Compute next valid actions for next state
        next_valid_actions_list = [get_valid_actions(env, next_obs[i]) for i in range(n_agents)]

        for i in range(n_agents):
            agents[i].store(
                obs[i],
                actions[i],
                rewards[i],
                next_obs[i],
                done[i],
                valid_actions_list[i],
                next_valid_actions_list[i]  # ✅ Store next valid actions
            )

            loss = agents[i].train_step()
            if loss is not None:
                episode_losses.append(loss)

            total_reward[i] += rewards[i]

        obs = next_obs
        complete_reward = sum(total_reward)

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
    })

print("\n✅ DQN Training Complete!")

torch.save(shared_policy_net.state_dict(), "shared_valid_dqn_model.pth")
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
        '''flat_obs = np.array(obs[i]).flatten()
        obs_tensor = torch.FloatTensor(flat_obs).unsqueeze(0)
        with torch.no_grad():
            q_vals = shared_policy_net(obs_tensor)


        action = int(q_vals.argmax())
        


        actions.append(action)
        print(f"Agent {i} action: {action}")'''
        valid_actions = get_valid_actions(env, obs[i])
        flat_obs = np.array(obs[i]).flatten()
        obs_tensor = torch.FloatTensor(flat_obs).unsqueeze(0)

        with torch.no_grad():
            q_values = shared_policy_net(obs_tensor).squeeze()
        
        q_values_np = q_values.cpu().numpy()
        masked_q = np.full_like(q_values_np, -np.inf)
        for a in valid_actions:
            masked_q[a] = q_values_np[a]
        best_action = int(np.argmax(masked_q))
        actions.append(best_action)

    obs, rewards, terminated, truncated, infos = env.step(actions)
    print(rewards)
    # print(info, "hello?")
    # terminated = [terminated] * n_agents if isinstance(terminated, bool) else terminated
    # truncated = [truncated] * n_agents if isinstance(truncated, bool) else truncated
    # done = [t or tr for t, tr in zip(terminated, truncated)]

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

# --- save GIF ---
imageio.mimsave("rware_dqn_valid_actions.gif", frames, fps=10)
print("🎥 Saved rware_dqn_valid_actions.gif")

input("🏁 Press Enter to exit...")
env.close()