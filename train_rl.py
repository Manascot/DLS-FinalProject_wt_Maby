# src/train_rl.py

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.distributions.normal import Normal
from torchvision import transforms
import matplotlib.pyplot as plt

# Make sure we can import utils.py from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils import make_noisy_lift_env  # instructor-provided
from src.models import PolicyNet


# ---------------------------------------------------------------------
# 1. Action sampling
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# 2. Movement-based shaped reward
# ---------------------------------------------------------------------

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

    # Success / height bonus (may rarely trigger if cube is lifted)
    cube_height = cube_pos_true[2]
    success_bonus = 0.0
    if cube_height > 0.10:  # lifted 10 cm
        success_bonus = 5.0

    # Small time penalty to encourage faster solutions
    time_penalty = -0.001

    shaped = move_reward + success_bonus + time_penalty
    return float(shaped), curr_dist_xy


# ---------------------------------------------------------------------
# 3. Single episode rollout
# ---------------------------------------------------------------------

def run_episode(policy, env, device, max_steps=150, gamma=0.99):
    """
    Run a single episode in one env, collect log_probs, shaped rewards,
    final cube height, and final XY distance.
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
    final_cube_height = 0.0
    final_xy_dist = 0.0

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

        # Track final cube height and xy distance at this step
        cube_pos_noisy = np.array(obs["cube_pos_noisy"], dtype=float)
        final_cube_height = cube_pos_noisy[2]
        if prev_dist_xy is not None:
            final_xy_dist = prev_dist_xy

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

    total_reward = sum(rewards)

    return log_probs, returns, total_reward, final_cube_height, final_xy_dist


# ---------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------

def train_agent(num_episodes=250, gamma=0.99, lr=1e-4, max_steps=150):
    """
    Trains the PolicyNet using REINFORCE on the noisy Lift env.
    Returns:
      policy,
      episode_returns: list of total shaped reward per episode
      episode_final_heights: list of final cube z per episode
      episode_final_xy_dists: list of final XY distances per episode
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNet().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_returns = []
    episode_final_heights = []
    episode_final_xy_dists = []

    for ep in range(1, num_episodes + 1):
        env = make_noisy_lift_env(add_noise=True, image_size=(256, 256))

        log_probs, returns, total_reward, final_height, final_dist = run_episode(
            policy, env, device, max_steps=max_steps, gamma=gamma
        )

        # Policy gradient loss
        loss = -(log_probs * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        episode_returns.append(total_reward)
        episode_final_heights.append(final_height)
        episode_final_xy_dists.append(final_dist)

        if ep % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            print(f"Episode {ep:4d} | Last 10 avg shaped return: {avg_return:.3f}")

    return policy, episode_returns, episode_final_heights, episode_final_xy_dists


# ---------------------------------------------------------------------
# 5. Plotting utilities
# ---------------------------------------------------------------------

def make_plots(episode_returns, heights, dists, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rewards = np.array(episode_returns)
    heights = np.array(heights)
    dists = np.array(dists)

    n = min(len(rewards), len(heights), len(dists))
    rewards = rewards[:n]
    heights = heights[:n]
    dists = dists[:n]
    episodes = np.arange(1, n + 1)

    # --- 1. Reward vs episode ---
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards, alpha=0.4, label="Episode shaped return")
    plt.xlabel("Episode")
    plt.ylabel("Shaped reward")
    plt.title("Reward vs Episode")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_vs_episode.png", dpi=150)
    plt.close()

    # --- 2. Final cube 'height' vs episode ---
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, heights, alpha=0.4, label="Final cube noisy z")
    plt.xlabel("Episode")
    plt.ylabel("Cube z at episode end (noisy)")
    plt.title("Final Cube z vs Episode")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "height_vs_episode.png", dpi=150)
    plt.close()

    # --- 3. Distance vs episode ---
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, dists, alpha=0.4, label="Final XY distance")
    plt.xlabel("Episode")
    plt.ylabel("XY distance (eef ↔ cube)")
    plt.title("Final XY Distance vs Episode")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "distance_vs_episode.png", dpi=150)
    plt.close()

    # --- 4. Combined reward vs height (dual axis) ---
    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Shaped Reward")
    ax1.plot(episodes, rewards, alpha=0.25, label="Reward")

    window = min(20, n)
    if window >= 2:
        kernel = np.ones(window) / window
        rewards_smooth = np.convolve(rewards, kernel, mode="valid")
        x_reward_smooth = np.arange(window, window + len(rewards_smooth))
        ax1.plot(x_reward_smooth, rewards_smooth, label=f"Reward MA ({window})")

    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Cube z at episode end (noisy)")
    ax2.plot(episodes, heights, alpha=0.25, label="Height", color="red")

    if window >= 2:
        heights_smooth = np.convolve(heights, kernel, mode="valid")
        x_height_smooth = np.arange(window, window + len(heights_smooth))
        ax2.plot(x_height_smooth, heights_smooth, label=f"Height MA ({window})", color="darkred")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Reward vs Final Cube z During Training")
    fig.tight_layout()
    plt.savefig(out_dir / "reward_vs_height_combined.png", dpi=150)
    plt.close()

    print(f"Saved plots to {out_dir}")


# ---------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------

def main():
    num_episodes = 250
    gamma = 0.99
    lr = 1e-4
    max_steps = 150

    policy, episode_returns, episode_final_heights, episode_final_xy_dists = train_agent(
        num_episodes=num_episodes,
        gamma=gamma,
        lr=lr,
        max_steps=max_steps,
    )

    # Save trained weights
    ckpt_path = PROJECT_ROOT / "policy_checkpoint.pth"
    torch.save(policy.state_dict(), ckpt_path)
    print(f"Saved trained policy to {ckpt_path}")

    # Make and save plots
    plots_dir = PROJECT_ROOT / "plots"
    make_plots(episode_returns, episode_final_heights, episode_final_xy_dists, plots_dir)


if __name__ == "__main__":
    main()
