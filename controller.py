# src/controller.py

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from IPython.display import Video as IPyVideo, display

# Ensure we can import utils and models
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils import (
    make_noisy_lift_env,
    init_frames_dir,
    save_frame,
    step_with_action,
    move_ee_to,
    is_lift_success,
)

from src.models import PolicyNet


# ===== 1. Model loading and optional download =====

MODEL_CKPT_PATH = PROJECT_ROOT / "policy_checkpoint.pth"
# TODO: put your real model download URL here once uploaded
MODEL_URL = "https://YOUR_DRIVE_OR_ONEDRIVE_LINK_HERE"


def download_model_if_needed(model_url: str, ckpt_path: Path):
    """
    Downloads the checkpoint from model_url if ckpt_path does not exist.
    You can adapt this to match your HW2 download code.
    """
    if ckpt_path.exists():
        print(f"Model checkpoint already exists at {ckpt_path}")
        return

    if not model_url or "http" not in model_url:
        raise ValueError(
            "MODEL_URL is not set. Please update MODEL_URL in controller.py "
            "or manually place policy_checkpoint.pth at project root."
        )

    print(f"Downloading model from {model_url} ...")
    import requests
    r = requests.get(model_url)
    r.raise_for_status()
    with open(ckpt_path, "wb") as f:
        f.write(r.content)
    print(f"Downloaded model to {ckpt_path}")


def load_policy(device):
    """
    Load PolicyNet and its weights from checkpoint.
    """
    download_model_if_needed(MODEL_URL, MODEL_CKPT_PATH)

    policy = PolicyNet().to(device)
    state_dict = torch.load(MODEL_CKPT_PATH, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"Loaded policy from {MODEL_CKPT_PATH}")
    return policy


# ===== 2. RL-corrected scripted controller for one episode =====

def run_rl_corrected_episode(policy, run_idx, add_noise=True):
    """
    Runs ONE episode of the RL-corrected scripted controller:
      1) Create a Lift env with (noisy) observations and random cube position.
      2) Use the policy once to correct the noisy cube position.
      3) Use the scripted controller (above -> down -> close -> lift -> hold).
      4) Save frames into a run-specific frames_dir.
      5) Return success flag, final cube height, and frames_dir.
    """
    env = make_noisy_lift_env(add_noise=add_noise, image_size=(256, 256))
    obs = env.reset()

    cam_name = "frontview"
    cam_key = f"{cam_name}_image"
    assert cam_key in obs, (
        f"Missing {cam_key} in obs. Make sure use_camera_obs=True and "
        f"camera_names includes '{cam_name}'."
    )

    # Starting cube position (noisy) for success evaluation
    cube_start_pos = np.asarray(obs["cube_pos_noisy"], dtype=float).copy()

    H, W = obs[cam_key].shape[:2]

    # RL correction
    device = next(policy.parameters()).device
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    img = obs[cam_key]                 # (H, W, 3)
    noisy_pos = obs["cube_pos_noisy"]  # (3,)

    img_tensor = transform(img).unsqueeze(0).to(device)
    pos_tensor = torch.FloatTensor(noisy_pos).unsqueeze(0).to(device)

    with torch.no_grad():
        mean, log_std = policy(img_tensor, pos_tensor)
    correction_xy = mean.cpu().numpy()[0]

    cube_pos_meas = np.asarray(noisy_pos, dtype=float).copy()
    cube_pos_corr = cube_pos_meas.copy()
    cube_pos_corr[:2] = cube_pos_meas[:2] + correction_xy

    print(f"[Run {run_idx}] Noisy cube pos:     {cube_pos_meas}")
    print(f"[Run {run_idx}] Corrected cube pos: {cube_pos_corr}")

    # Frames dir for this run
    frames_dir = f"frames_run_{run_idx}"
    init_frames_dir(frames_dir)

    action_dim = env.action_dim
    print(f"[Run {run_idx}] action_dim =", action_dim)

    frame_id = 0
    frame_id = save_frame(obs, cam_key, frame_id, frames_dir=frames_dir)

    # Waypoints from corrected cube pos
    above_height = 0.15
    grasp_height = 0.02
    lift_height = 0.25

    target_above = cube_pos_corr.copy()
    target_above[2] += above_height

    target_grasp = cube_pos_corr.copy()
    # target_grasp[2] += grasp_height  # if you want

    target_lift = cube_pos_corr.copy()
    target_lift[2] += lift_height

    # Phases
    print(f"[Run {run_idx}] Phase 0: open gripper")
    action_open = np.zeros(action_dim, dtype=float)
    action_open[-1] = -1.0
    obs, frame_id = step_with_action(
        env,
        action_open,
        n_steps=20,
        obs=obs,
        cam_key=cam_key,
        frame_id=frame_id,
        frames_dir=frames_dir,
    )

    print(f"[Run {run_idx}] Phase 1: move above cube (corrected target)")
    obs, frame_id = move_ee_to(
        env,
        obs,
        target_pos_or_fn=target_above,
        gripper=-1.0,
        steps=100,
        action_dim=action_dim,
        cam_key=cam_key,
        frame_id=frame_id,
        frames_dir=frames_dir,
        kp=8.0,
        ki=0.0,
        kd=1.0,
        max_delta=0.1,
    )

    print(f"[Run {run_idx}] Phase 2: move down to grasp height (corrected target)")
    obs, frame_id = move_ee_to(
        env,
        obs,
        target_pos_or_fn=target_grasp,
        gripper=-1.0,
        steps=250,
        action_dim=action_dim,
        cam_key=cam_key,
        frame_id=frame_id,
        frames_dir=frames_dir,
        kp=[10.0, 10.0, 12.0],
        ki=0.0,
        kd=0.2,
        max_delta=0.1,
    )

    print(f"[Run {run_idx}] Phase 3: close gripper")
    action_close = np.zeros(action_dim, dtype=float)
    action_close[-1] = 1.0
    obs, frame_id = step_with_action(
        env,
        action_close,
        n_steps=40,
        obs=obs,
        cam_key=cam_key,
        frame_id=frame_id,
        frames_dir=frames_dir,
    )

    print(f"[Run {run_idx}] Phase 4: lift cube (corrected target)")
    obs, frame_id = move_ee_to(
        env,
        obs,
        target_pos_or_fn=target_lift,
        gripper=1.0,
        steps=200,
        action_dim=action_dim,
        cam_key=cam_key,
        frame_id=frame_id,
        frames_dir=frames_dir,
        kp=10.0,
        ki=0.0,
        kd=1.0,
        max_delta=0.1,
    )

    print(f"[Run {run_idx}] Phase 5: hold")
    obs, frame_id = step_with_action(
        env,
        np.zeros(action_dim, dtype=float),
        n_steps=40,
        obs=obs,
        cam_key=cam_key,
        frame_id=frame_id,
        frames_dir=frames_dir,
    )

    print(f"[Run {run_idx}] RL-corrected scripted rollout finished. Frames saved to {frames_dir}")

    # Success check
    success = is_lift_success(
        obs,
        cube_start_pos=cube_start_pos,
        min_lift=0.10,
        max_xy_shift=0.10,
        use_noisy=True,
    )

    final_cube_height = float(np.asarray(obs["cube_pos_noisy"], dtype=float)[2])
    print(f"[Run {run_idx}] Lift success: {success}, final cube height (noisy): {final_cube_height:.3f}")

    return success, final_cube_height, frames_dir


# ===== 3. Test on 10 envs + best video =====

def save_video(frames_dir, output_filename, fps=20):
    """
    Create an MP4 from frames using ffmpeg CLI.
    Expects frames like frame_0000.png in frames_dir.
    """
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(".png")
    ])
    if not frame_files:
        print(f"[save_video] No frames in {frames_dir}")
        return None

    cmd = (
        f"ffmpeg -y -framerate {fps} "
        f"-i {frames_dir}/frame_%04d.png "
        f"-c:v libx264 -pix_fmt yuv420p {output_filename}"
    )
    print("[save_video] Running:", cmd)
    os.system(cmd)

    if os.path.exists(output_filename):
        print(f"[save_video] Video saved to {output_filename}")
        return output_filename
    else:
        print(f"[save_video] Failed to create {output_filename}")
        return None


def test_controller_on_10_envs(policy, num_envs=10, video_name="best_result_demo.mp4"):
    results = []

    for i in range(1, num_envs + 1):
        print("=" * 60)
        print(f"Running RL-corrected controller on environment {i}/{num_envs}")
        success, final_height, frames_dir = run_rl_corrected_episode(policy, run_idx=i)
        results.append({
            "idx": i,
            "success": success,
            "final_height": final_height,
            "frames_dir": frames_dir,
        })

    # Success rate
    success_count = sum(1 for r in results if r["success"])
    success_rate = 100.0 * success_count / num_envs
    print("=" * 60)
    print(f"Success rate over {num_envs} environments: {success_rate:.1f}% ({success_count}/{num_envs})")

    # Best episode: prefer successful with highest height; otherwise highest height overall
    successful_runs = [r for r in results if r["success"]]
    if successful_runs:
        best_run = max(successful_runs, key=lambda r: r["final_height"])
        print(f"Best run (SUCCESS): env {best_run['idx']} with final height {best_run['final_height']:.3f}")
    else:
        best_run = max(results, key=lambda r: r["final_height"])
        print(f"No successful runs. Best ATTEMPT: env {best_run['idx']} with final height {best_run['final_height']:.3f}")

    best_frames_dir = best_run["frames_dir"]

    video_file = save_video(best_frames_dir, video_name, fps=20)
    if isinstance(video_file, str) and os.path.exists(video_file):
        print(f"Best result video saved to {video_file}")
        display(IPyVideo(video_file, embed=True, width=640))
    else:
        print("Failed to create best result video. video_file =", video_file)

    return success_rate, results, video_file


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(device)

    success_rate, results, video_file = test_controller_on_10_envs(
        policy,
        num_envs=10,
        video_name="best_result_demo.mp4",
    )

    print("=" * 60)
    print(f"FINAL TEST SUCCESS RATE: {success_rate:.1f}%")
    print(f"DEMO VIDEO: {video_file}")


if __name__ == "__main__":
    main()
