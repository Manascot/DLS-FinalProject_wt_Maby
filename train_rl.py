# src/train_rl.py

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torchvision import transforms

# Make sure we can import utils.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils import make_noisy_lift_env  # instructor-provided

from src.models import PolicyNet


def select_action(policy, img_tensor, pos_tensor):
    """
    img_tensor: (1, 3, 64, 64)
    pos_tensor: (1, 3)
    Returns:
      action (np array, shape (2,)),
      log_prob (torch scalar)
    """
    mean, log_std = policy(img_tensor, pos_tensor)
    std = torch.exp(log_std)

    dist = Normal(mean, std)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(dim=-1)  # sum over action dims

    return action.detach().cpu().numpy()[0], log_prob


def compute_step_reward(obs, prev_dist_xy=None):
    """
    Movement-based shaped reward:
      - Positive reward if the gripper moves closer to the cube in XY.
      - Negative reward if it moves farther away.
      - Small bonus if cube is lifted above threshold.
      - Small time penalty per step.
    """
    gripper_pos = obs.get("robot0_eef_pos", None)   # (3,)
    cube_pos_true = obs.get("cube_pos", None)       # (3,)

    # If we can't access positions, no shaping
    if gripper_pos is None or cube_pos_true is None:
        return 0.0, prev_dist_xy

    gripper_pos = np.array(gripper_pos, dtype=float)
    cube_pos_true = np.array(cube_pos_true, dtype=float)

    # Distance in XY plane
    gripper_xy = gripper_pos[:2]
    cube_xy = cube_pos_true[:2]
    curr_dist_xy = np.linalg.norm(gripper_xy - cube_xy)

    # Movement reward: compare previous distance to current
    if prev_dist_xy is None:
        move_reward = 0.0
    else:
        delta = prev_dist_xy - curr_dist_xy  # >0 if we got closer
        move_reward = 5.0 * delta            # scale factor

    # Success / height bonus
    cube_height = cube_pos_true[2]
    success_bonus = 0.0
    if cube_height > 0.10:  # lifted 10 cm
        success_bonus = 5.0

    # Small time penalty to encourage faster solutions (per step)
    time_penalty = -0.001

    shaped = move_reward + success_bonus + time_penalty
    return float(shaped), curr_dist_xy


def run_episode(policy, env, device, max_steps=150, gamma=0.99):
    """
    Run a single episode in one env, collect log_probs and shaped rewards.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    obs = env.reset()
    rewards = []
    log_probs = []

    prev_dist_xy = None

    for t in range(max_steps):
        img = obs["frontview_image"]
        noisy_pos = obs["cube_pos_noisy"]

        # Prepare tensors
        img_tensor = transform(img).unsqueeze(0).to(device)
        pos_tensor = torch.FloatTensor(noisy_pos).unsqueeze(0).to(device)

        # Get 2D correction from policy
        action_xy, log_prob = select_action(policy, img_tensor, pos_tensor)

        # Build full action (env expects action_dim, e.g., 7)
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[:2] = np.clip(action_xy, -1.0, 1.0)

        obs, env_reward, done, info = env.step(action)

        # Movement-based shaped reward
        r, prev_dist_xy = compute_step_reward(obs, prev_dist_xy)
        rewards.append(r)
        log_probs.append(log_prob)

        if done:
            break

    # Compute discounted returns
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8 + 1e-12)

    log_probs = torch.stack(log_probs)

    return log_probs, returns, sum(rewards)


def train_agent(num_episodes=250, gamma=0.99, lr=1e-4, max_steps=150):
    """
    Trains the PolicyNet using REINFORCE on the noisy Lift env.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNet().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_returns = []

    for ep in range(1, num_episodes + 1):
        env = make_noisy_lift_env(add_noise=True, image_size=(256, 256))

        log_probs, returns, total_reward = run_episode(
            policy, env, device, max_steps=max_steps, gamma=gamma
        )

        # Policy gradient loss
        loss = -(log_probs * returns).sum()

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

        optimizer.step()

        episode_returns.append(total_reward)

        if ep % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            print(f"Episode {ep:4d} | Last 10 avg shaped return: {avg_return:.3f}")

    return policy


def main():
    num_episodes = 1250
    gamma = 0.99
    lr = 1e-4
    max_steps = 150

    policy = train_agent(
        num_episodes=num_episodes,
        gamma=gamma,
        lr=lr,
        max_steps=max_steps,
    )

    # Save trained weights for later download / use
    ckpt_path = PROJECT_ROOT / "policy_checkpoint.pth"
    torch.save(policy.state_dict(), ckpt_path)
    print(f"Saved trained policy to {ckpt_path}")
    print("Upload this file to your online drive and put the link in README.md.")


if __name__ == "__main__":
    main()
