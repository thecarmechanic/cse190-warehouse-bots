
import gymnasium as gym
import rware

import copy
import random
import time
import pickle
import numpy as np

# other imports
import matplotlib.pyplot as plt
import imageio
from pyglet import image

# --- Set up environment ---
env = gym.make("rware-tiny-2ag-v2")
obs = env.reset()
print(obs)

frames = []

# --- Q-learning config ---
n_agents = 2
n_actions = 5
alpha = 0.1         # learning rate
gamma = 0.95        # discount
epsilon = 1.0       # exploration rate
epsilon_decay = 0.999
epsilon_min = 0.05
num_episodes = 2000
max_steps = 300     # per episode

# --- Define small custom layout ---
layout = '''
x.x
.g.
x.x
'''

# --- Create environment with custom layout ---
env = gym.make("rware:rware-tiny-2ag-v2", layout=layout)
obs, _ = env.reset()

# --- Initialize Q-tables ---
Q_tables = [{} for _ in range(n_agents)]

def get_state_key(obs_vector):
    # Discretize and compress observation
    return tuple(np.round(obs_vector, 1)) # [::4]

def choose_action(agent_id, obs):
    state = get_state_key(obs)
    if np.random.rand() < epsilon:
        return env.action_space[agent_id].sample()
    q_vals = Q_tables[agent_id].get(state, np.zeros(n_actions))
    return int(np.argmax(q_vals))

# --- Training loop ---
for episode in range(num_episodes):
    obs, _ = env.reset()
    done = [False] * n_agents
    total_reward = [0] * n_agents

    for step in range(max_steps):
        actions = [choose_action(i, obs[i]) for i in range(n_agents)]
        next_obs, rewards, terminated, truncated, _ = env.step(actions)

        if not isinstance(terminated, (list, tuple)):
            terminated = [terminated] * n_agents
        if not isinstance(truncated, (list, tuple)):
            truncated = [truncated] * n_agents

        done = [t or tr for t, tr in zip(terminated, truncated)]

        for i in range(n_agents):
            s = get_state_key(obs[i])
            s_next = get_state_key(next_obs[i])

            Q_tables[i].setdefault(s, np.zeros(n_actions))
            Q_tables[i].setdefault(s_next, np.zeros(n_actions))

            best_next = np.max(Q_tables[i][s_next])
            Q_tables[i][s][actions[i]] += alpha * (rewards[i] + gamma * best_next - Q_tables[i][s][actions[i]])

            total_reward[i] += rewards[i]

        obs = next_obs
        if all(done):
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 500 == 0:
        sizes = [len(q) for q in Q_tables]
        print(f"Episode {episode+1}: epsilon={epsilon:.3f}, Q-table sizes={sizes}, reward={total_reward}")

# # --- Save Q-tables (optional) ---
# with open("q_tables.pkl", "wb") as f:
#     pickle.dump(Q_tables, f)

print("\n✅ Training complete!")

# --- Evaluation ---
obs, _ = env.reset()
done = [False] * n_agents

for step in range(max_steps):
    actions = []
    for i in range(n_agents):
        state = get_state_key(obs[i])
        q_vals = Q_tables[i].get(state, np.zeros(n_actions))
        actions.append(int(np.argmax(q_vals)))

    obs, rewards, terminated, truncated, _ = env.step(actions)

    if not isinstance(terminated, (list, tuple)):
        terminated = [terminated] * n_agents
    if not isinstance(truncated, (list, tuple)):
        truncated = [truncated] * n_agents

    done = [t or tr for t, tr in zip(terminated, truncated)]

    env.render()
    # Capture the current frame from the pyglet window
    if hasattr(env.unwrapped, "renderer") and hasattr(env.unwrapped.renderer, "window"):
        win = env.unwrapped.renderer.window
        win.switch_to()
        win.dispatch_events()
        buffer = image.get_buffer_manager().get_color_buffer()
        img_data = buffer.get_image_data()
        frame = np.frombuffer(img_data.get_data('RGB', buffer.width * 3), dtype=np.uint8)
        frame = frame.reshape((buffer.height, buffer.width, 3))
        frame = np.flipud(frame)  # flip vertically
        frames.append(frame)
    time.sleep(0.1)

    if all(done):
        break

# Save frames as GIF
imageio.mimsave("rware_eval.gif", frames, fps=10)
print("✅ Saved rware_eval.gif")

input("\n🏁 Finished. Press Enter to exit...")
env.close()

