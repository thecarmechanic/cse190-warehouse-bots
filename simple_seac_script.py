import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*The reward returned by `step\(\)` must be a float.*"
)

import argparse
import collections
import random
import time

import gymnasium as gym


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Make sure you have installed RWARE:
#   pip install rware
# and that ‘rware’ is registered as a Gymnasium env.
# (If you use PyPI’s “rware”, the import is just gym.make("rware:…"))


# -----------------------------
#  1. Hyperparameters & Config
# -----------------------------
parser = argparse.ArgumentParser(description="Simplified SEAC on RWARE")
parser.add_argument(
    "--env-name",
    type=str,
    default="rware:rware-tiny-4ag-v2",  # 4-agent tiny map
    help="Gym environment ID for RWARE (e.g. rware:rware-tiny-4ag-v2)",
)
parser.add_argument("--num-agents", type=int, default=4, help="Number of agents")
parser.add_argument(
    "--num-episodes", type=int, default=500, help="Total training episodes"
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
    "--batch-size",
    type=int,
    default=32,
    help="Mini-batch size for updates (not used in this simple code)",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed"
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
        # x: [batch, obs_dim]
        logits = self.net(x)
        return logits  # to be used with Categorical(logits=...)


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
        # x: [batch, obs_dim]
        v = self.net(x).squeeze(-1)
        return v  # [batch]


# -----------------------------
#  3. Helper Functions
# -----------------------------
def preprocess_obs(obs):
    """
    The RWARE partial observation is a small grid with multiple channels.
    We simply flatten into a 1D float vector.
    Input: obs as a NumPy array (e.g. shape [3,3,?])
    Output: 1D torch tensor
    """
    return torch.from_numpy(obs.astype(np.float32).ravel())


def compute_returns(rewards, dones, gamma):
    """
    Compute discounted returns for a sequence.
    Args:
        rewards: list of floats
        dones: list of bools (episode-done flags)
        gamma: discount factor
    Returns:
        returns: torch.tensor of shape [len(rewards)]
    """
    returns = []
    R = 0.0
    for r, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0.0  # reset if episode ended
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


# -----------------------------
#  4. Main Training Loop
# -----------------------------
def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create RWARE environment
    env = gym.make(args.env_name)
    # env.reset() returns (obs_tuple, info); we ignore info here
    obs_tuple, _ = env.reset(seed=args.seed)
    # obs_tuple is a tuple of length num_agents; each obs is an array
    # Let's infer each obs_dim
    obs_dim = preprocess_obs(obs_tuple[0]).shape[0]
    act_dim = env.action_space[0].n  # discrete actions per agent

    # Create one PolicyNet & one ValueNet per agent
    policies = []
    values = []
    pi_opts = []
    v_opts = []
    for i in range(args.num_agents):
        pi_net = PolicyNet(obs_dim, act_dim, args.hidden_size)
        v_net = ValueNet(obs_dim, args.hidden_size)
        policies.append(pi_net)
        values.append(v_net)
        pi_opts.append(optim.Adam(pi_net.parameters(), lr=args.lr_pi))
        v_opts.append(optim.Adam(v_net.parameters(), lr=args.lr_v))

    # For logging
    episode_rewards = [0.0 for _ in range(args.num_agents)]
    all_return_logs = collections.deque(maxlen=100)

    start_time = time.time()

    for ep in range(1, args.num_episodes + 1):
        # Per-episode storage per agent
        agent_buffers = [
            {
                "obs": [],
                "acts": [],
                "logps": [],
                "vals": [],
                "rews": [],
                "dones": [],
            }
            for _ in range(args.num_agents)
        ]

        # Reset environment at episode start
        obs_tuple, _ = env.reset()
        done_flags = [False] * args.num_agents

        for step in range(args.rollout_length):
            # 1. For each agent, select action
            actions = []
            logps = []
            vals = []
            for i in range(args.num_agents):
                obs_i = preprocess_obs(obs_tuple[i]).unsqueeze(0)  # [1, obs_dim]
                logits = policies[i](obs_i)  # [1, act_dim]
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()  # Tensor([action_index])
                logp = dist.log_prob(a)  # [1]
                v = values[i](obs_i)  # [1]

                actions.append(a.item())
                logps.append(logp.squeeze(0))
                vals.append(v.squeeze(0))

                # Store obs, logp, val
                agent_buffers[i]["obs"].append(obs_i.squeeze(0))
                agent_buffers[i]["logps"].append(logp.squeeze(0))
                agent_buffers[i]["vals"].append(v.squeeze(0))

            # 2. Step environment
            next_obs_tuple, reward_tuple, done_all, truncated, info = env.step(
                tuple(actions)
            )

            env.render()

            # RWARE returns reward per agent as list of floats
            # RWARE's done_all is a single bool, but we treat it per-agent
            done_flags = [done_all] * args.num_agents

            # 3. Store rewards & done flags
            for i in range(args.num_agents):
                agent_buffers[i]["acts"].append(actions[i])
                agent_buffers[i]["rews"].append(reward_tuple[i])
                agent_buffers[i]["dones"].append(done_flags[i])
                episode_rewards[i] += reward_tuple[i]

            obs_tuple = next_obs_tuple
            if all(done_flags):
                break

        # ------------------------
        #  4. Post-episode Updates
        # ------------------------
        # For each agent, compute its returns & advantages from its own data
        returns_per_agent = []
        advs_per_agent = []
        for i in range(args.num_agents):
            rews = agent_buffers[i]["rews"]
            dones = agent_buffers[i]["dones"]
            vals_tensor = torch.stack(agent_buffers[i]["vals"])
            returns = compute_returns(rews, dones, args.gamma)  # [T]
            returns_per_agent.append(returns)

            # advantage = R_t - V(s_t)
            adv = returns - vals_tensor.detach()
            advs_per_agent.append(adv)

        # For each agent i, build combined loss
        for i in range(args.num_agents):
            # Zero gradients
            pi_opts[i].zero_grad()
            v_opts[i].zero_grad()

            # Gather i’s own data
            logps_i = torch.stack(agent_buffers[i]["logps"])  # [T]
            adv_i = advs_per_agent[i]  # [T]

            # (1) Own actor loss
            L_actor_own = -torch.mean(logps_i * adv_i)

            # (2) Shared actor loss from others’ rollouts
            L_actor_shared = torch.tensor(0.0)
            if args.num_agents > 1:
                for j in range(args.num_agents):
                    if j == i:
                        continue
                    logps_j = []
                    obs_list_j = []
                    acts_j = agent_buffers[j]["acts"]  # [T]
                    for t in range(len(agent_buffers[j]["obs"])):
                        obs_j_t = agent_buffers[j]["obs"][t].unsqueeze(0)
                        logits_i = policies[i](obs_j_t)
                        dist_i = torch.distributions.Categorical(logits=logits_i)
                        a_j_t = torch.tensor([acts_j[t]])
                        logp_i_on_j = dist_i.log_prob(a_j_t).squeeze(0)
                        logps_j.append(logp_i_on_j)
                    logps_j = torch.stack(logps_j)  # [T]
                    adv_j = advs_per_agent[j]  # [T]
                    L_actor_shared = L_actor_shared - torch.mean(
                        logps_j * adv_j
                    )

                # Average over other agents
                L_actor_shared = L_actor_shared / (args.num_agents - 1)

            # (3) Critic loss: use *all* returns to train V_i
            returns_all = torch.cat([returns_per_agent[j] for j in range(args.num_agents)])
            obs_all = torch.cat(
                [
                    torch.stack(agent_buffers[j]["obs"])
                    for j in range(args.num_agents)
                ],
                dim=0,
            )  # shape [N*T, obs_dim]
            # Value predictions V_i(s) for all agents’ obs
            V_preds_all = values[i](obs_all)
            L_critic = torch.mean((returns_all - V_preds_all) ** 2)

            # Total losses
            loss_pi = L_actor_own + args.lambda_shared * L_actor_shared
            loss_v = L_critic

            # Backpropagate
            loss_pi.backward()
            loss_v.backward()
            pi_opts[i].step()
            v_opts[i].step()

        # ------------------------
        #  5. Logging & Saving
        # ------------------------
        avg_ep_reward = sum(episode_rewards) / args.num_agents
        all_return_logs.append(avg_ep_reward)
        if ep % 10 == 0:
            avg100 = np.mean(all_return_logs) if len(all_return_logs) > 0 else 0.0
            print(
                f"Episode {ep:4d} | AvgReward: {avg_ep_reward:6.2f} | "
                f"100-ep Avg: {avg100:6.2f} | Time: {time.time()-start_time:5.1f}s"
            )
        # Reset per-episode reward counters
        episode_rewards = [0.0 for _ in range(args.num_agents)]

        # Save a checkpoint every 100 episodes
        if ep % 100 == 0:
            torch.save(
                {
                    f"pi_{i}": policies[i].state_dict()
                    for i in range(args.num_agents)
                },
                f"seac_pi_ep{ep}.pth",
            )
            torch.save(
                {
                    f"v_{i}": values[i].state_dict()
                    for i in range(args.num_agents)
                },
                f"seac_v_ep{ep}.pth",
            )

    env.close()
    print("Training completed.")


if __name__ == "__main__":
    main()
