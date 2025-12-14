# Ankle Gait Stability In Humanoids

This project was done as a part of the EECE 5552 (Assistive Robotics) course at Northeastern University.

## Project Overview:

This project investigates ankle load dynamics and gait stability in humanoid robots and a human musculoskeletal model under increasing external loads. Using a combination of Model Predictive Control (MPC), reinforcement learning (RL), and physiologically inspired PD control, we analyze how different control strategies and embodiments influence ankle torque demand, balance, and robustness. The study compares two humanoid platforms (Unitree H1 and G1) against a MyoSuite musculoskeletal model to draw insights relevant to assistive locomotion and ankle exoskeleton design.

## Simulation Videos

### H1 – No Load (MPC vs RL)
### G1 vs H1 – 70 kg Load (RL)
### G1 – RL with Increasing Load (30 kg vs 70 kg)

## Repository Structure

Data files/
Contains all logged CSV files for ankle torques, joint states, and CoM data across platforms (H1, G1, and MyoSuite) and load conditions.

~Plots/~
Includes all generated figures used in the analysis, such as torque time-series, peak torque vs load plots, scaling trends, and CoM trajectories.

Analysis/
Python scripts for data preprocessing, filtering, statistical analysis, and figure generation.
