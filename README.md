# Ankle Gait Stability In Humanoids

This project was done as a part of the EECE 5552 (Assistive Robotics) course at Northeastern University.

## Project Overview:

This project investigates ankle load dynamics and gait stability in humanoid robots and a human musculoskeletal model under increasing external loads. Using a combination of Model Predictive Control (MPC), reinforcement learning (RL), and physiologically inspired PD control, we analyze how different control strategies and embodiments influence ankle torque demand, balance, and robustness. The study compares two humanoid platforms (Unitree H1 and G1) against a MyoSuite musculoskeletal model to draw insights relevant to assistive locomotion and ankle exoskeleton design.

## Simulation Videos

### H1 – No Load (MPC vs RL)

Demonstrates differences between model-based and learning-based control on the H1 humanoid under nominal conditions.

https://github.com/user-attachments/assets/c0851d9a-0409-4242-a661-a436f2c9d8ae

### G1 vs H1 – 70 kg Load (RL)

Side-by-side comparison highlighting the effect of ankle morphology (single-axis vs biaxial) on balance and stability under extreme loading.

https://github.com/user-attachments/assets/1f489f69-b0c0-464e-9ecb-21c0b1437f23

### G1 – RL with Increasing Load (30 kg vs 70 kg)

Shows how a pre-trained RL policy adapts to increasing payloads, including visible changes in ankle torque and center-of-mass sway.

https://github.com/user-attachments/assets/e79ec366-e4f8-49c8-81f4-c647df13931e

## Repository Structure

Data files/

Contains all logged CSV files for ankle torques, joint states, and CoM data across platforms (H1, G1, and MyoSuite) and load conditions.

Plots/

Includes all generated figures used in the analysis, such as torque time-series, peak torque vs load plots, scaling trends, and CoM trajectories.

Analysis/

Python scripts for data preprocessing, filtering, statistical analysis, and figure generation.

## References:

https://github.com/iit-DLSLab/mpx

https://github.com/unitreerobotics/unitree_rl_gym 

https://github.com/MyoHub/myosuite 

https://github.com/leggedrobotics/legged_gym 



