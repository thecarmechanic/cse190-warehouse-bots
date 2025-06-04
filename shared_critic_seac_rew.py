import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*The reward returned by step\(\) must be a float.*"
)

import argparse
import collections
import random
import time
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# -----------------------------
#  1. Hyperparameters & Config
# -----------------------------
parser = argparse.ArgumentParser(
    description="Improved SEAC on RWARE with Shaping, Entropy, Adv Norm, Grad Clip, and Artifacts"
)
parser.add_argument(
    "--env-name",
    type=str,
    default="rware:rware-tiny-2ag-v2",  # two-agent tiny map by default
    help="Gym environment ID for RWARE (e.g. rware:rware-tiny-2ag-v2)",
)
parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
parser.add_argument(
    "--num-episodes",
    type=int,
    default=2000,
    help="Total training episodes",
)
parser.add_argument(
    "--rollout-length", type=int, default=200, help="Max steps per episode"
)
parser.add_argument(
    "--gamma", type=float, default=0.99, help="Discount factor"
)
parser.add_argument(
    "--lr-pi", type=float, default=1e-3, help="Policy network learning rate"
)
parser.add_argument(
    "--lr-v", type=float, default=1e-3, help="Value network learning rate"
)
parser.add_argument(
    "--lambda-shared",
    type=float,
    default=0.5,
    help="Weight for shared-experience actor loss",
)
parser.add_argument(
    "--hidden-size", type=int, default=64, help="Hidden layer size in networks"
)
parser.add_argument(
    "--entropy-coef",
    type=float,
    default=0.01,
    help="Entropy regularization coefficient",
)
parser.add_argument(
    "--shaping-coef",
    type=float,
    default=0.0,
    help="Reward-shaping coefficient (potential-based). Set to 0.0 to disable.",
)
parser.add_argument(
    "--max-grad-norm",
    type=float,
    default=0.5,
    help="Max norm for gradient clipping",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed"
)
parser.add_argument(
    "--save-dir",
    type=str,
    default="checkpoints",
    help="Directory where checkpoints and plots will be saved",
)
args = parser.parse_args()


# -----------------------------
#  2. Network Definitions
# -----------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits  # for Categorical(logits=...)


class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        v = self.net(x).squeeze(-1)
        return v  # [batch]


# -----------------------------
#  3. Helper Functions
# -----------------------------
def preprocess_obs(obs):
    """
    Flatten a partial observation (H×W×C) → 1D tensor [obs_dim].
    """
    return torch.from_numpy(obs.astype(np.float32).ravel())


def manhattan_to_nearest_item(agent_pos, item_positions):
    """
    Compute Manhattan distance from agent_pos to nearest item (list of (x,y)).
    Returns 0.0 if no items remain.
    """
    if item_positions is None or len(item_positions) == 0:
        return 0.0
    ax, ay = agent_pos
    d_min = float("inf")
    for (ix, iy) in item_positions:
        d = abs(ax - ix) + abs(ay - iy)
        if d < d_min:
            d_min = d
    return float(d_min)


def compute_returns(rewards, dones, gamma):
    """
    Compute discounted returns for a sequence (potentially shaped rewards).
    """
    returns = []
    R = 0.0
    for r, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0.0
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


# -----------------------------
#  4. Main Training Loop
# -----------------------------
def main():
    # 4.1 Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 4.2 Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # 4.3 Make RWARE env
    env = gym.make(args.env_name)
    obs_tuple, _ = env.reset(seed=args.seed)
    # Determine obs_dim by flattening
    obs_dim = preprocess_obs(obs_tuple[0]).shape[0]
    act_dim = env.action_space[0].n  # discrete actions per agent

    # 4.4 Instantiate networks & optimizers
    policies = []
    pi_opts = []
    for i in range(args.num_agents):
        pi_net = PolicyNet(obs_dim, act_dim, args.hidden_size)
        policies.append(pi_net)
        pi_opts.append(optim.Adam(pi_net.parameters(), lr=args.lr_pi))

    # One shared critic
    value_net = ValueNet(obs_dim, args.hidden_size)
    v_opt = optim.Adam(value_net.parameters(), lr=args.lr_v)

    # 4.5 Logging buffers
    episode_rewards = [0.0 for _ in range(args.num_agents)]
    all_return_logs = collections.deque(maxlen=100)

    reward_history_avg = []
    reward_history_agents = [[] for _ in range(args.num_agents)]
    actor_loss_history = []
    critic_loss_history = []
    entropy_history = []

    start_time = time.time()

    for ep in range(1, args.num_episodes + 1):
        # 4.6.1 Create per-episode buffers
        agent_buffers = [
            {
                "obs": [],    # list of tensors [obs_dim]
                "acts": [],   # ints
                "logps": [],  # tensors
                "ents": [],   # tensors
                "vals": [],   # tensors
                "rews": [],   # floats (shaped)
                "dones": [],  # bools
            }
            for _ in range(args.num_agents)
        ]

        # 4.6.2 Reset and initialize shaping distances
        obs_tuple, _ = env.reset()
        prev_distances = [float("inf")] * args.num_agents
        done_flags = [False] * args.num_agents

        # 4.6.3 Rollout

        for step in range(args.rollout_length):
            if step == 0:
                info = {}
            actions = []

            # 1) Each agent picks an action
            for i in range(args.num_agents):
                obs_i = preprocess_obs(obs_tuple[i]).unsqueeze(0)  # [1, obs_dim]

                # Evaluate value with no_grad
                with torch.no_grad():
                    v_i = value_net(obs_i)  # [1]

                logits_i = policies[i](obs_i)  # [1, act_dim]
                action_mask = info.get("action_mask", [None] * args.num_agents)[i]
                if action_mask is not None:
                    mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=logits_i.device)
                    logits_i = logits_i.clone()  # avoid in-place ops
                    logits_i[0, ~mask_tensor] = -1e9  # batch dimension at index 0
                dist_i = torch.distributions.Categorical(logits=logits_i)
                a_i = dist_i.sample()
                if action_mask is not None and not action_mask[a_i.item()]:
                    print(f"[EP {ep}] Agent {i} chose INVALID action {a_i.item()}")

                logp_i = dist_i.log_prob(a_i).squeeze(0)  # scalar
                ent_i = dist_i.entropy().squeeze(0)       # scalar

                actions.append(a_i.item())

                # Store
                agent_buffers[i]["obs"].append(obs_i.squeeze(0))
                agent_buffers[i]["logps"].append(logp_i)
                agent_buffers[i]["ents"].append(ent_i)
                agent_buffers[i]["vals"].append(v_i.squeeze(0))

            # 2) Step env
            next_obs_tuple, reward_tuple, done_all, truncated, info = env.step(tuple(actions))

            # No rendering for speed
            env.render()

            done_flags = [done_all] * args.num_agents

            # 3) Reward shaping (potential-based)
            item_positions = info.get("item_pos", None)    # list of (x,y)
            agent_positions = info.get("agent_pos", None)  # list of (x,y)
            delivery_positions = info.get("delivery_pos", None)
            carrying = info.get("carrying", [False] * args.num_agents)
            new_distances = []
            shaped_rewards = []

            for i in range(args.num_agents):
                raw_r = reward_tuple[i]
                if agent_positions is None:
                    new_distances.append(prev_distances[i])
                    shaped_rewards.append(raw_r)
                    continue

                # Potential based on item or delivery zone
                if carrying[i] and delivery_positions:
                    # Moving toward delivery zone
                    d_new = manhattan_to_nearest_item(agent_positions[i], delivery_positions)
                else:
                    # Moving toward pickup zone
                    d_new = manhattan_to_nearest_item(agent_positions[i], item_positions)

                dist_bonus = args.shaping_coef * (prev_distances[i] - d_new)

                # Add bonus for specific events
                pickup_bonus = 0.0
                delivery_bonus = 0.0

                # Detect pickup
                if info.get("just_picked_up", [False] * args.num_agents)[i]:
                    pickup_bonus = 0.1  # Encourage pickup

                # Detect delivery
                if info.get("just_delivered", [False] * args.num_agents)[i]:
                    delivery_bonus = 0.3  # Stronger reward for correct delivery

                # Total shaped reward
                total_reward = raw_r + dist_bonus + pickup_bonus + delivery_bonus
                new_distances.append(d_new)
                shaped_rewards.append(total_reward)

            prev_distances = new_distances[:]


            # 4) Store shaped rewards & dones
            for i in range(args.num_agents):
                agent_buffers[i]["acts"].append(actions[i])
                agent_buffers[i]["rews"].append(shaped_rewards[i])
                agent_buffers[i]["dones"].append(done_flags[i])
                episode_rewards[i] += shaped_rewards[i]

            obs_tuple = next_obs_tuple
            if all(done_flags):
                break

        # ------------------------
        #  4.6.4 Compute returns & advantages
        # ------------------------
        returns_per_agent = []
        advs_per_agent = []

        for i in range(args.num_agents):
            rews = agent_buffers[i]["rews"]
            dones = agent_buffers[i]["dones"]
            vals_tensor = torch.stack(agent_buffers[i]["vals"])  # [T]

            returns_i = compute_returns(rews, dones, args.gamma)  # [T]
            returns_per_agent.append(returns_i)

            adv_i = returns_i - vals_tensor.detach()  # [T]
            adv_i = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)
            advs_per_agent.append(adv_i)

        # ------------------------
        #  4.6.5 Actor & Critic Updates
        # ------------------------
        actor_losses = []
        avg_entropies = []

        # (A) Actor updates per agent
        for i in range(args.num_agents):
            pi_opts[i].zero_grad()

            logps_i = torch.stack(agent_buffers[i]["logps"])  # [T]
            ents_i = torch.stack(agent_buffers[i]["ents"])    # [T]
            adv_i = advs_per_agent[i]                         # [T]

            # Own actor loss + entropy bonus
            L_actor_own = -torch.mean(logps_i * adv_i) - args.entropy_coef * torch.mean(ents_i)

            # Shared actor loss
            L_actor_shared = torch.tensor(0.0, dtype=torch.float32)
            if args.num_agents > 1:
                for j in range(args.num_agents):
                    if j == i:
                        continue
                    logps_on_j = []
                    acts_j = agent_buffers[j]["acts"]
                    for t, obs_j_t in enumerate(agent_buffers[j]["obs"]):
                        obs_j_t_b = obs_j_t.unsqueeze(0)
                        logits_i_on_j = policies[i](obs_j_t_b)
                        dist_i_on_j = torch.distributions.Categorical(logits=logits_i_on_j)
                        a_j_t = torch.tensor([acts_j[t]])
                        logp_i_on_j = dist_i_on_j.log_prob(a_j_t).squeeze(0)
                        logps_on_j.append(logp_i_on_j)
                    logps_on_j = torch.stack(logps_on_j)  # [T_j]
                    adv_j = advs_per_agent[j]             # [T_j]
                    L_actor_shared += -torch.mean(logps_on_j * adv_j)
                L_actor_shared = L_actor_shared / (args.num_agents - 1)

            loss_pi_i = L_actor_own + args.lambda_shared * L_actor_shared
            actor_losses.append(loss_pi_i.item())
            avg_entropies.append(torch.mean(ents_i).item())

            loss_pi_i.backward()
            torch.nn.utils.clip_grad_norm_(policies[i].parameters(), args.max_grad_norm)
            pi_opts[i].step()

        # (B) Critic update (shared)
        v_opt.zero_grad()
        obs_all_list = []
        for j in range(args.num_agents):
            obs_all_list.append(torch.stack(agent_buffers[j]["obs"]))  # [T_j, obs_dim]
        obs_all = torch.cat(obs_all_list, dim=0)  # [sum(T_j), obs_dim]
        returns_all = torch.cat(returns_per_agent, dim=0)         # [sum(T_j)]

        V_preds_all = value_net(obs_all)  # [sum(T_j)]
        L_critic = torch.mean((returns_all - V_preds_all) ** 2)
        critic_loss_history.append(L_critic.item())

        L_critic.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), args.max_grad_norm)
        v_opt.step()

        # ------------------------
        #  4.6.6 Logging
        # ------------------------
        avg_ep_reward = sum(episode_rewards) / args.num_agents
        reward_history_avg.append(avg_ep_reward)
        for i in range(args.num_agents):
            reward_history_agents[i].append(episode_rewards[i])

        actor_loss_history.append(float(np.mean(actor_losses)))
        entropy_history.append(float(np.mean(avg_entropies)))

        all_return_logs.append(avg_ep_reward)
        if ep % 10 == 0:
            avg100 = np.mean(all_return_logs) if len(all_return_logs) > 0 else 0.0
            elapsed = time.time() - start_time
            print(
                f"Episode {ep:4d} | "
                f"AvgReward: {avg_ep_reward:6.2f} | "
                f"100-ep Avg: {avg100:6.2f} | "
                f"ActorLoss: {actor_loss_history[-1]:6.4f} | "
                f"CriticLoss: {critic_loss_history[-1]:6.4f} | "
                f"Entropy: {entropy_history[-1]:.4f} | "
                f"Elapsed: {elapsed:5.1f}s"
            )
        if ep % 100 == 0:
            pickups = info.get("valid_pickups", 0)
            deliveries = info.get("valid_deliveries", 0)
            print(f"  ↳ Pickups: {pickups} | Deliveries: {deliveries}")

        episode_rewards = [0.0 for _ in range(args.num_agents)]

        # 4.6.7 Checkpoint every 100 episodes
        if ep % 100 == 0:
            for i in range(args.num_agents):
                torch.save(
                    policies[i].state_dict(),
                    os.path.join(args.save_dir, f"agent{i}_pi_ep{ep}.pth")
                )
            torch.save(
                value_net.state_dict(),
                os.path.join(args.save_dir, f"value_shared_ep{ep}.pth")
            )

    env.close()
    print("Training completed.")

    # -----------------------------
    #  5. Artifact Generation & Plotting
    # -----------------------------
    os.makedirs(args.save_dir, exist_ok=True)

    # Convert to numpy
    avg_array = np.array(reward_history_avg, dtype=np.float32)
    actor_loss_array = np.array(actor_loss_history, dtype=np.float32)
    critic_loss_array = np.array(critic_loss_history, dtype=np.float32)
    entropy_array = np.array(entropy_history, dtype=np.float32)
    agent_arrays = [np.array(reward_history_agents[i], dtype=np.float32) for i in range(args.num_agents)]

    # Save raw data
    np.save(os.path.join(args.save_dir, "reward_history_avg.npy"), avg_array)
    for i in range(args.num_agents):
        np.save(os.path.join(args.save_dir, f"reward_history_agent{i}.npy"), agent_arrays[i])
    np.save(os.path.join(args.save_dir, "actor_loss_history.npy"), actor_loss_array)
    np.save(os.path.join(args.save_dir, "critic_loss_history.npy"), critic_loss_array)
    np.save(os.path.join(args.save_dir, "entropy_history.npy"), entropy_array)

    # Plot 1: Average Reward + Rolling Mean
    plt.figure(figsize=(8, 5))
    episodes = np.arange(1, len(avg_array) + 1)
    plt.plot(episodes, avg_array, color="C0", alpha=0.4, label="Avg Reward")
    if len(avg_array) >= 100:
        rolling = np.convolve(avg_array, np.ones(100) / 100, mode="valid")
        plt.plot(episodes[99:], rolling, color="C0", linewidth=2, label="Rolling 100-ep Mean")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("SEAC on RWARE (2 agents) – Average Reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "avg_reward_curve.png"))
    plt.close()

    # Plot 2: Per-Agent Rewards
    plt.figure(figsize=(8, 5))
    for i in range(args.num_agents):
        plt.plot(episodes, agent_arrays[i], alpha=0.6, label=f"Agent {i} Reward")
    plt.xlabel("Episode")
    plt.ylabel("Per-Agent Reward")
    plt.title("SEAC on RWARE (2 agents) – Per-Agent Rewards")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "per_agent_reward_curve.png"))
    plt.close()

    # Plot 3: Loss Curves
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, actor_loss_array, color="C1", alpha=0.6, label="Actor Loss")
    plt.plot(episodes, critic_loss_array, color="C2", alpha=0.6, label="Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("SEAC on RWARE (2 agents) – Loss Curves")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "loss_curves.png"))
    plt.close()

    # Plot 4: Entropy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(episodes, entropy_array, color="C3", alpha=0.7, label="Avg Entropy")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.title("SEAC on RWARE (2 agents) – Policy Entropy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "entropy_curve.png"))
    plt.close()

    # -----------------------------
    #  6. Optional: Save GIF of Trained Agents
    # -----------------------------
    from PIL import Image

    frames = []
    obs_tuple, _ = env.reset()
    done_flags = [False] * args.num_agents

    for _ in range(args.rollout_length):
        frame = env.render(mode="rgb_array")
        frames.append(Image.fromarray(frame))

        actions = []
        for i in range(args.num_agents):
            obs_i = preprocess_obs(obs_tuple[i]).unsqueeze(0)
            with torch.no_grad():
                logits = policies[i](obs_i)

                # 🔒 Mask invalid actions
                action_mask = info.get("action_mask", [None] * args.num_agents)[i]
                if action_mask is not None:
                    logits[~torch.tensor(action_mask, dtype=torch.bool)] = -1e9

                # 🧠 Deterministic action selection
                a = torch.argmax(logits, dim=-1)

            actions.append(a.item())

        obs_tuple, _, done_all, truncated, info = env.step(tuple(actions))

        if done_all:
            break

    gif_path = os.path.join(args.save_dir, "trained_agents.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,  # ms per frame
        loop=0
    )
    print(f"🎥 Saved final rollout GIF to: {gif_path}")

    print(f"Saved plots and data into '{args.save_dir}/'.")


if __name__ == "__main__":
    main()