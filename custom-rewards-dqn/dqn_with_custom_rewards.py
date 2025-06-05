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
epsilon_decay = 0.9995 # 0.9995
epsilon_min = 0.05
num_episodes = 10000 # 5000
max_steps = 300
batch_size = 64
target_update_freq = 10
replay_capacity = 10000
max_reward = 50
progress_check = 100    # how often we want to print episode progress in training loop

frames = []

# --- layout ---
small_layout = """
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

# --- environment ---
env = gym.make("rware:rware-tiny-2ag-v2", layout=small_layout)
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

# --- helper functions ---
# manhattan heuristic - returns distance to closest target
def manhattan(pos, target):
    return abs(pos[0] - target[0]) + abs(pos[1] - target[1]) 

# A* pathfinding for reward shaping
import heapq

def astar(grid_size, start, goal, blocked, heuristic):
    open_set = [(0 + heuristic(start, goal), 0, start)]
    visited = set()
    g_cost = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            return cost

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if (0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1]
                and neighbor not in blocked):
                new_cost = cost + 1
                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
    return float('inf')

# --- initialize agents ---
agents = [Agent(i, shared_policy_net, shared_target_net, shared_optimizer, shared_memory) for i in range(n_agents)]

episode_stats = []  # before the loop

# --- training loop ---
for episode in range(num_episodes):
    episode_losses = []
    obs, _ = env.reset()
    total_reward = [0] * n_agents
    total_shaped_reward = [0] * n_agents
    done = [False] * n_agents
    visited_states = [set() for _ in range(n_agents)] # save visited states for each agent to access by index
    
    for i in range(n_agents):
        visited_states[i].add(tuple(obs[i]))
    
    for step in range(max_steps):
        actions = [agent.select_action(obs[i], epsilon) for i, agent in enumerate(agents)]
        next_obs, rewards, terminated, truncated, info = env.step(actions)

        terminated = [terminated] * n_agents if isinstance(terminated, bool) else terminated
        truncated = [truncated] * n_agents if isinstance(truncated, bool) else truncated
        done = [t or tr for t, tr in zip(terminated, truncated)]

        for i in range(n_agents):
            # --- custom reward shaping ---
            raw_reward = rewards[i]
            shaped_reward = raw_reward
            # if any(tuple(next_obs[i]) in ag_states for ag_states in visited_states.values()): # penalize exploring the state already explored by both agents
            #     # next_obs[i].flatten() in visited_states[i]: # penalize exploring the state already explored by the same agent
            #     shaped_reward -= 0.1 
            # print the q values for all the actions

            # reward function 1:
            # if all(obs[i] == next_obs[i]): # penalize for taking invalid actions -- no state change
            #     shaped_reward -= 0.1
            
            # if raw_reward == 1:
            #     shaped_reward += 1
            
            # reward function 2:
            # incentivize explorations & penalize revisiting the same state
            state = tuple(obs[i])
            if raw_reward == 0:
                if state in visited_states[i]:
                    shaped_reward -= 0.02 # light penalty # med = -0.01
                    # NEW: additional idle state dis-incentive to stop unfruitful random rotations (for medium layout)
                    if actions[i] in (2,3):
                        shaped_reward -= 0.01
                else:
                    visited_states[i].add(state)
                    # shaped_reward += 0.05
            
            # penalize for taking invalid actions and no-op-- no state change
            if all(obs[i] == next_obs[i]):
                shaped_reward -= 0.1 
            
            # # punish no-op
            # if actions[i] == 0:
            #     shaped_reward -= 0.05
            
            # bonus to successful delivery
            if raw_reward == 1:
                shaped_reward += 2
                # NEW: encourage repeat delivery
                if (obs[i][2] > 0.5) and (next_obs[i][2] < 0.5):  # dropped a shelf
                    shaped_reward += 0.2
            
            # reward function 3: adds on to reward 2 -- not working very well unfort
            # time step penalty to disincentivize idle states
            shaped_reward -= 0.005

            # euclidean heuristic to incentivize carrying a shelf to a goal state
            curr_loc = obs[i][0:2]
            next_loc = next_obs[i][0:2]
            valid_shelves = [tuple((int(shelf.x), int(shelf.y))) for shelf in env.unwrapped.shelfs if shelf in env.unwrapped.request_queue]
            goals = env.unwrapped.goals

            if (obs[i][2] < 0.5):
                closest = min([manhattan(curr_loc, shelf) - manhattan(next_loc, shelf) for shelf in valid_shelves])
                if closest > 0:
                    shaped_reward += 0.005 * closest # medium = 0.02
                # if euclidean(curr_loc, valid_shelves) > euclidean(next_loc, valid_shelves):
                #     shaped_reward += 0.05
            
            elif (obs[i][2] > 0.5):
                closest =min([manhattan(curr_loc, goal) - manhattan(next_loc, goal) for goal in goals])
                if closest > 0:
                    shaped_reward += 0.005 * closest # medium = 0.02
                # if euclidean(curr_loc, goals) > euclidean(next_loc, goals):
                #     shaped_reward += 0.05
            
            # incentivize requested shelf pickup -- unfortunately this seems to incentivize arbitrary shelf toggling
            # if (obs[i][2] < 0.5) and (next_obs[i][2] > 0.5):
            #     shaped_reward += 0.1
            
            shaped_reward = np.clip(shaped_reward, -2.0, 2.0) #NEW
            # --- save and train ---
            agents[i].store(obs[i], actions[i], shaped_reward, next_obs[i], done[i])
            loss = agents[i].train_step()
            if loss is not None:
                episode_losses.append(loss)
            total_reward[i] += rewards[i]
            total_shaped_reward[i] += shaped_reward
            
        obs = next_obs
        complete_reward = 0
        for r in total_reward:
            complete_reward += r

        if all(done) or complete_reward >= max_reward:
            break
    
    avg_loss = np.mean(episode_losses) if episode_losses else 0.0
    total_reward_sum = sum(total_reward)
    avg_reward_per_agent = total_reward_sum / n_agents
    total_shaped_reward_sum = sum(total_shaped_reward)
    avg_shaped_reward_per_agent = total_shaped_reward_sum / n_agents
    
    
    # --- epsilon decay ---
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # --- update target net? ---
    if (episode + 1) % target_update_freq == 0:
        shared_target_net.load_state_dict(shared_policy_net.state_dict())
    
    # --- print progress? ---
    if (episode + 1) % progress_check == 0:
        print(f"Episode {episode+1}, epsilon={epsilon:.3f}, "
              f"total_reward={total_reward_sum}, avg_reward={avg_reward_per_agent:.2f}, "
              f"total_shaped_reward={total_shaped_reward_sum}, avg_shaped_reward={avg_shaped_reward_per_agent:.2f}, "
              f"avg_loss={avg_loss:.4f}")
    
    episode_stats.append({
        "episode": episode + 1,
        "epsilon": epsilon,
        "total_reward": total_reward_sum,
        "avg_reward": avg_reward_per_agent,
        "total_shaped_reward": total_shaped_reward_sum,
        "avg_shaped_reward": avg_shaped_reward_per_agent,
        "avg_loss": avg_loss
    })

print("\n✅ DQN Training Complete!")

torch.save(shared_policy_net.state_dict(), "shared_rewards_dqn_model.pth")
print("✅ Model saved to 'shared_dqn_rewards_model.pth'")

with open("./small_reward3_episode_stats.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=episode_stats[0].keys())
    writer.writeheader()
    writer.writerows(episode_stats)

# --- evaluation ---
obs, _ = env.reset()
done = [False] * n_agents
rewards_log = []
total_reward = 0

for step in range(max_steps):
    actions = []
    for i in range(n_agents):
        flat_obs = np.array(obs[i]).flatten()
        obs_tensor = torch.tensor(flat_obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_vals = shared_policy_net(obs_tensor).detach().cpu().numpy()
        action = int(q_vals.argmax())
        actions.append(action)
        print(f"Agent {i} action: {action}")

    next_obs, rewards, terminated, truncated, infos = env.step(actions)
    # print(info, "hello?")
    terminated = [terminated] * n_agents if isinstance(terminated, bool) else terminated
    truncated = [truncated] * n_agents if isinstance(truncated, bool) else truncated
    done = [t or tr for t, tr in zip(terminated, truncated)]
    for r in rewards:
        total_reward += r
    rewards_log.append({"step":step + 1, "total reward": total_reward})

    with open("./small_reward3_eval_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rewards_log[0].keys())
        writer.writeheader()
        writer.writerows(rewards_log)

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
    print(f"TOTAL REWARD: {total_reward}")
    obs = next_obs

# --- save GIF ---
imageio.mimsave("small_rware_dqn_reward3_eval.gif", frames, fps=10)
print("🎥 Saved small_rware_dqn_reward3_eval.gif")

input("🏁 Press Enter to exit...")
env.close()