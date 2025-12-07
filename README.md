# RL-Based Noise-Robust Lift Controller

## 1. Codebase Overview (Short Description)

This codebase implements a **hybrid controller** for a noisy robotic lifting task (Robosuite “Lift”):

- A **neural network policy** (trained with reinforcement learning) that takes:
  - A front-view RGB image of the scene, and  
  - A noisy estimate of the cube (object) position,  
  and outputs a **2D correction** to the cube’s XY position.

- A **scripted waypoint controller** that:
  - Moves the end-effector above the (corrected) cube position,
  - Moves down to grasp height,
  - Closes the gripper, and
  - Lifts and holds the cube.

The RL policy acts as a **noise-correction module** that improves the cube localization, while the scripted controller handles the low-level motion. The codebase also includes a training script for reinforcement learning and a testing script that evaluates the controller on multiple random environments and records a demo video.

### Directory structure

```text
.
├── utils.py              # Provided helpers: env creation and low-level control
├── src/
│   ├── __init__.py
│   ├── models.py         # PolicyNet (neural network for XY correction)
│   ├── train_rl.py       # RL training (REINFORCE) with shaped reward
│   └── controller.py     # Trained controller + evaluation + video generation
└── README.md             # This file


2. How to Run the Code
2.1. Requirements

You will need (typical homework environment):

Python 3.8+

PyTorch

Torchvision

Robosuite

Requests (for downloading the model)

ffmpeg (command-line tool, for making MP4 videos)

Example installation (adjust to your environment):

pip install torch torchvision requests
# robosuite is usually provided / set up via the course environment
# ffmpeg must be installed at the system level (e.g. via apt, brew, or conda)


Make sure utils.py (provided by the instructor) is in the project root alongside src/.

2.2. Training the RL Policy (Reinforcement Learning)

The RL training script is src/train_rl.py. It:

Uses a movement-based shaped reward:

Positive when the end-effector moves closer to the cube in the XY-plane,

Negative when it moves farther away,

Bonus when the cube is lifted above a height threshold.

Uses REINFORCE with gradient clipping for stability.

Trains on the noisy Lift environment for a specified number of episodes.

Saves the trained model weights to policy_checkpoint.pth in the project root.

From the project root, run:

python -m src.train_rl


This will:

Initialize and train PolicyNet.

Print the training progress, including the average shaped return over the last 10 episodes.

Save the trained policy weights to:

policy_checkpoint.pth


You should then upload this file to your online drive (see Section 3).

2.3. Using the Trained Controller (Test-Time Evaluation + Video)

The controller script is src/controller.py. It:

Loads the trained model policy_checkpoint.pth:

Either from the project root, or

Downloads it automatically from the URL you specify (see Section 3).

Creates 10 random simulation environments with:

Random object (cube) locations, and

Observation noise.

In each environment, runs the RL-corrected scripted controller:

The neural network predicts a correction to the noisy cube position.

The scripted controller moves:

Above the corrected cube position,

Down to grasp,

Closes the gripper,

Lifts and holds the cube.

Computes the success rate over the 10 environments.

Selects the best episode:

If there are any successful runs: choose the one with the highest final cube height.

If there are no successes: choose the attempt with the highest final cube height.

Records a demo video (best_result_demo.mp4) from the selected episode.

To run the controller (from the project root):

python -m src.controller


This will:

Ensure the checkpoint is available (download if needed),

Print something like:

Success rate over 10 environments: XX.X% (k/10)
FINAL TEST SUCCESS RATE: XX.X%
DEMO VIDEO: best_result_demo.mp4


And create best_result_demo.mp4 in the project root, showing the best attempt by the controller.

You should submit this demo video along with your report.


3. Trained Model Download Links

3.1. Main Trained Policy

This is the model used by src/controller.py.

Checkpoint filename: policy_checkpoint.pth

Download link:
[https://your-storage-link-here.com/path/to/policy_checkpoint.pth](https://drive.google.com/drive/folders/1A8-LHTKUhYz2iqSPYrfrYeZFEQrTEDhm?usp=sharing)


