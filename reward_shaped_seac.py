import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*The reward returned by `step\(\)` must be a float.*"
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
import imageio   # for GIF creation


# -----------------------------
#  1. Hyperparameters & Config
# -----------------------------
parser = argparse.ArgumentParser(
    description="SEAC on RWARE (with IS terms) + GIF Recording + Artifact Plots"
)
parser.add_argument(
    "--env-name",
    type=str,
    default="rware:rware-tiny-2ag-v2",
    help="Gym environment ID for RWARE (e.g. rware:rware-tiny-2ag-v2)",
)
parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
parser.add_argument(
    "--num-episodes",
    type=int,
    default=10000,
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
    default=1.0,
    help="Weight for shared-experience loss (λ in the paper)",
)
parser.add_argument(
    "--hidden-size", type=int, default=64, help="Hidden layer size in networks"
)
parser.add_argument(
    "--entropy-coef",
    type=float,
    default=0.05,
    help="Entropy regularization coefficient",
)
parser.add_argument(
    "--shaping-coef",
    type=float,
    default=0.1,
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
    help="Directory where checkpoints, plots, and GIF will be saved",
)
args = parser.parse_args()


# -----------------------------
#  2. Network Definitions
# -----------------------------
class PolicyNet(nn.Module):
    """
    Simple MLP policy. Input: flattened partial observation (obs_dim,). Output: logits over actions.
    """
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
        # x: [batch, obs_dim]
        logits = self.net(x)
        return logits  # for Categorical(logits=...)


class ValueNet(nn.Module):
    """
    Simple MLP value network. Input: flattened partial observation (obs_dim,). Output: scalar V(s).
    """
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
        # x: [batch, obs_dim]
        v = self.net(x).squeeze(-1)
        return v  # [batch]


# -----------------------------
#  3. Helper Functions
# -----------------------------
def preprocess_obs(obs):
    """
    Flatten a partial observation (H×W×C) → 1D torch tensor [obs_dim].
    """
    return torch.from_numpy(obs.astype(np.float32).ravel())


def manhattan_to_nearest_item(agent_pos, item_positions):
    """
    Compute Manhattan distance from agent_pos to nearest item. If no items, return inf.
    """
    if item_positions is None or len(item_positions) == 0:
        return float("inf")
    ax, ay = agent_pos
    return float(min(abs(ax - ix) + abs(ay - iy) for ix, iy in item_positions))


def compute_returns(rewards, dones, gamma):
    """
    Compute discounted returns (R_t) for a sequence of (shaped) rewards & done flags.
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
    # 4.1: Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 4.2: Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # 4.3: Create RWARE environment and infer obs_dim, act_dim
    env = gym.make(args.env_name)
    obs_tuple, info = env.reset(seed=args.seed)
    obs_dim = preprocess_obs(obs_tuple[0]).shape[0]
    act_dim = env.action_space[0].n  # discrete actions per agent

    # 4.4: Instantiate networks & optimizers
    policies = []
    pi_opts = []
    for i in range(args.num_agents):
        pi_net = PolicyNet(obs_dim, act_dim, args.hidden_size)
        policies.append(pi_net)
        pi_opts.append(optim.Adam(pi_net.parameters(), lr=args.lr_pi))

    # One shared critic
    value_net = ValueNet(obs_dim, args.hidden_size)
    v_opt = optim.Adam(value_net.parameters(), lr=args.lr_v)

    # 4.5: Logging structures
    episode_rewards = [0.0 for _ in range(args.num_agents)]
    all_return_logs = collections.deque(maxlen=100)

    reward_history_avg = []
    reward_history_agents = [[] for _ in range(args.num_agents)]
    actor_loss_history = []
    critic_loss_history = []
    entropy_history = []

    start_time = time.time()

    # 4.6: Training loop over episodes
    for ep in range(1, args.num_episodes + 1):
        # 4.6.1: Per-episode buffers
        agent_buffers = [
            {
                "obs": [],    # list of tensors [obs_dim]
                "acts": [],   # list of ints
                "logps": [],  # list of torch scalars
                "ents": [],   # list of torch scalars
                "vals": [],   # list of torch scalars
                "rews": [],   # list of shaped rewards (floats)
                "dones": [],  # list of bools
            }
            for _ in range(args.num_agents)
        ]

        # 4.6.2: Reset & initialize shaping distances
        obs_tuple, info = env.reset(seed=args.seed + ep)
        agent_positions = info.get("agent_pos", None)
        item_positions = info.get("item_pos", None)
        prev_distances = []
        for i in range(args.num_agents):
            if agent_positions is None:
                prev_distances.append(float("inf"))
            else:
                prev_distances.append(
                    manhattan_to_nearest_item(agent_positions[i], item_positions)
                )
        done_flags = [False] * args.num_agents

        # 4.6.3: Rollout up to rollout_length
        for step in range(args.rollout_length):
            actions = []

            # (1) Each agent picks an action
            for i in range(args.num_agents):
                obs_i = preprocess_obs(obs_tuple[i]).unsqueeze(0)  # [1, obs_dim]

                # Evaluate value with no_grad to avoid building graph
                with torch.no_grad():
                    v_i = value_net(obs_i)  # [1]

                logits_i = policies[i](obs_i)  # [1, act_dim]
                dist_i = torch.distributions.Categorical(logits=logits_i)
                a_i = dist_i.sample()           # [1]
                logp_i = dist_i.log_prob(a_i).squeeze(0)  # scalar
                ent_i = dist_i.entropy().squeeze(0)       # scalar

                actions.append(a_i.item())

                # Store per-step data
                agent_buffers[i]["obs"].append(obs_i.squeeze(0))  # [obs_dim]
                agent_buffers[i]["logps"].append(logp_i)
                agent_buffers[i]["ents"].append(ent_i)
                agent_buffers[i]["vals"].append(v_i.squeeze(0))

            # (2) Step environment
            next_obs_tuple, reward_tuple, done_all, truncated, info = env.step(tuple(actions))
            done_flags = [done_all] * args.num_agents

            # (3) Compute shaped rewards (potential-based)
            item_positions = info.get("item_pos", None)
            agent_positions = info.get("agent_pos", None)

            new_distances = []
            for i in range(args.num_agents):
                if agent_positions is None:
                    new_distances.append(prev_distances[i])
                else:
                    new_distances.append(
                        manhattan_to_nearest_item(agent_positions[i], item_positions)
                    )

            shaped_rewards = []
            for i in range(args.num_agents):
                raw_r = float(reward_tuple[i])
                if prev_distances[i] != float("inf") and new_distances[i] != float("inf"):
                    dist_bonus = args.shaping_coef * (prev_distances[i] - new_distances[i])
                else:
                    dist_bonus = 0.0
                shaped_r = raw_r + dist_bonus
                shaped_rewards.append(shaped_r)

            # (4) Store shaped rewards & dones
            for i in range(args.num_agents):
                agent_buffers[i]["acts"].append(actions[i])
                agent_buffers[i]["rews"].append(shaped_rewards[i])
                agent_buffers[i]["dones"].append(done_flags[i])
                episode_rewards[i] += shaped_rewards[i]

            obs_tuple = next_obs_tuple
            prev_distances = new_distances[:]
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

            adv_i = returns_i - vals_tensor.detach()
            # Normalize advantage
            adv_i = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)
            advs_per_agent.append(adv_i)

        # ------------------------
        #  4.6.5 SEAC Actor & Critic Updates
        # ------------------------
        actor_losses = []
        avg_entropies = []

        # (A) Actor update per agent (with IS terms)
        for i in range(args.num_agents):
            pi_opts[i].zero_grad()

            # Gather agent i's own on-policy terms
            logps_i = torch.stack(agent_buffers[i]["logps"])  # [T]
            ents_i = torch.stack(agent_buffers[i]["ents"])    # [T]
            adv_i = advs_per_agent[i]                         # [T]

            # Own actor loss = −E[ logπ_i(a_i|o_i)*adv_i ] − entropy bonus
            L_actor_own = -torch.mean(logps_i * adv_i) - args.entropy_coef * torch.mean(ents_i)

            # Now add shared experience from every other agent j≠i (off-policy)
            L_actor_shared = torch.tensor(0.0, dtype=torch.float32)
            for j in range(args.num_agents):
                if j == i:
                    continue

                adv_j = advs_per_agent[j]  # [T_j]
                logps_on_j = []
                for t, obs_j_t in enumerate(agent_buffers[j]["obs"]):
                    obs_j_t_b = obs_j_t.unsqueeze(0)  # [1, obs_dim]
                    a_j_t = torch.tensor([agent_buffers[j]["acts"][t]])

                    # π_i(a_j^t|o_j^t)
                    logits_i_on_j = policies[i](obs_j_t_b)  # [1, act_dim]
                    dist_i_on_j = torch.distributions.Categorical(logits=logits_i_on_j)
                    logp_i_on_j = dist_i_on_j.log_prob(a_j_t).squeeze(0)

                    # π_j(a_j^t|o_j^t)
                    with torch.no_grad():
                        logits_j = policies[j](obs_j_t_b)
                        dist_j = torch.distributions.Categorical(logits=logits_j)
                        logp_j_on_j = dist_j.log_prob(a_j_t).squeeze(0)

                    # Importance weight = π_i / π_j
                    weight = torch.exp(logp_i_on_j - logp_j_on_j)
                    logps_on_j.append(weight * logp_i_on_j)

                if len(logps_on_j) > 0:
                    logps_on_j = torch.stack(logps_on_j)  # [T_j]
                    L_actor_shared += -torch.mean(logps_on_j * adv_j)

            # Average the shared loss over (num_agents − 1)
            if args.num_agents > 1:
                L_actor_shared = L_actor_shared / (args.num_agents - 1)

            # Total actor loss for agent i
            loss_pi_i = L_actor_own + args.lambda_shared * L_actor_shared
            actor_losses.append(loss_pi_i.item())
            avg_entropies.append(torch.mean(ents_i).item())

            # Backpropagate actor loss
            loss_pi_i.backward()
            torch.nn.utils.clip_grad_norm_(policies[i].parameters(), args.max_grad_norm)
            pi_opts[i].step()

        # (B) Critic update (shared across all agents, with IS terms)
        v_opt.zero_grad()

        # Build a big batch of all agents' observations
        obs_all_list = []
        returns_all_list = []
        for j in range(args.num_agents):
            obs_j = torch.stack(agent_buffers[j]["obs"])  # [T_j, obs_dim]
            returns_j = returns_per_agent[j]              # [T_j]
            obs_all_list.append(obs_j)
            returns_all_list.append(returns_j)

        obs_all = torch.cat(obs_all_list, dim=0)       # [sum(T_j), obs_dim]
        returns_all = torch.cat(returns_all_list, dim=0)  # [sum(T_j)]

        # Critic predicts V(s) for that big batch
        V_preds_all = value_net(obs_all)  # [sum(T_j)]

        # Base MSE loss (on-policy) = (returns_all − V_preds_all)^2
        L_critic = torch.mean((returns_all - V_preds_all) ** 2)

        # Now add off-policy terms
        off_critic_loss = torch.tensor(0.0, dtype=torch.float32)
        pair_count = 0

        for i in range(args.num_agents):
            for j in range(args.num_agents):
                if i == j:
                    continue
                pair_count += 1
                for t, obs_j_t in enumerate(agent_buffers[j]["obs"]):
                    obs_j_t_b = obs_j_t.unsqueeze(0)  # [1, obs_dim]
                    with torch.no_grad():
                        if t + 1 < len(agent_buffers[j]["obs"]):
                            next_obs_j = agent_buffers[j]["obs"][t + 1].unsqueeze(0)
                            V_next = value_net(next_obs_j).item()
                        else:
                            V_next = 0.0

                    r_j_t = agent_buffers[j]["rews"][t]

                    # π_i(a_j^t | o_j^t)
                    a_j_t = torch.tensor([agent_buffers[j]["acts"][t]])
                    logits_i_on_j = policies[i](obs_j_t_b)
                    dist_i_on_j = torch.distributions.Categorical(logits=logits_i_on_j)
                    logp_i_on_j = dist_i_on_j.log_prob(a_j_t).squeeze(0)

                    # π_j(a_j^t | o_j^t)
                    with torch.no_grad():
                        logits_j_on_j = policies[j](obs_j_t_b)
                        dist_j_on_j = torch.distributions.Categorical(logits=logits_j_on_j)
                        logp_j_on_j = dist_j_on_j.log_prob(a_j_t).squeeze(0)

                    w_ijt = torch.exp(logp_i_on_j - logp_j_on_j)

                    # TD target: y_j = r_j + γ V(next_obs)
                    y_jt = r_j_t + args.gamma * V_next
                    V_pred_jt = value_net(obs_j_t_b).squeeze(0)
                    se = (V_pred_jt - y_jt) ** 2

                    off_critic_loss += w_ijt * se

        if pair_count > 0:
            off_critic_loss = off_critic_loss / pair_count

        # Total critic loss = on-policy + λ * off-policy
        loss_v = L_critic + args.lambda_shared * off_critic_loss
        critic_loss_history.append(loss_v.item())

        loss_v.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), args.max_grad_norm)
        v_opt.step()

        # ------------------------
        #  4.6.6 Logging per episode
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

        # Reset per-episode rewards
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

    # Convert lists to numpy arrays
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
    plt.savefig(os.path.join(args.save-dir), "avg_reward_curve.png")
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
    plt.savefig(os.path.join(args.save-dir), "per_agent_reward_curve.png")
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
    plt.savefig(os.path.join(args.save-dir), "loss_curves.png")
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
    plt.savefig(os.path.join(args.save-dir), "entropy_curve.png")
    plt.close()

    print(f"Saved plots and data into '{args.save-dir}/'.")

    # -----------------------------
    #  6. Record a short evaluation GIF
    # -----------------------------
    eval_env = gym.make(args.env_name)
    final_ep = args.num_episodes
    for i in range(args.num_agents):
        ckpt_path_pi = os.path.join(args.save-dir, f"agent{i}_pi_ep{final_ep}.pth")
        policies[i].load_state_dict(torch.load(ckpt_path_pi))

    frames = []
    obs_tuple, _ = eval_env.reset(seed=args.seed + 1234)
    done_flags = [False] * args.num_agents

    for _ in range(args.rollout_length):
        frame = eval_env.render(mode="rgb_array")
        frames.append(frame)

        pk_actions = []
        for i in range(args.num_agents):
            obs_i = preprocess_obs(obs_tuple[i]).unsqueeze(0)
            logits = policies[i](obs_i)
            a_i = torch.argmax(logits, dim=-1).item()
            pk_actions.append(a_i)

        obs_tuple, _, done_all, _, _ = eval_env.step(tuple(pk_actions))
        done_flags = [done_all] * args.num_agents
        if all(done_flags):
            break

    gif_path = os.path.join(args.save-dir, "trained_agents.gif")
    imageio.mimsave(gif_path, frames, fps=20)
    eval_env.close()
    print(f"Saved evaluation GIF to '{gif_path}'.")


if __name__ == "__main__":
    main()
