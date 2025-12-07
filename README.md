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
