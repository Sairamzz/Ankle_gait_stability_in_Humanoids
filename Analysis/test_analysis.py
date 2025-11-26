#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data

df_torque = pd.read_csv("/home/sairam/NU/Assistive/Results/csv_files/h1_torques_labeled_2.csv")
df_qpos   = pd.read_csv("/home/sairam/NU/Assistive/Results/csv_files/h1_qpos_labeled_2.csv")
df_qvel   = pd.read_csv("/home/sairam/NU/Assistive/Results/csv_files/h1_qvel_labeled_2.csv")

print("Loaded torque:", df_torque.shape)
print("Loaded qpos:", df_qpos.shape)
print("Loaded qvel:", df_qvel.shape)

# Extract ankle-related signals

ankle = pd.DataFrame({
    "t": df_qpos["time_step"],
    # angles
    "L_angle": df_qpos["left_ankle"],
    "R_angle": df_qpos["right_ankle"],
    # velocities
    "L_vel": df_qvel["left_ankle_vel"],
    "R_vel": df_qvel["right_ankle_vel"],
    # torques
    "L_tau": df_torque["left_ankle"],
    "R_tau": df_torque["right_ankle"]
})

print("\nAnkle data shape:", ankle.shape)
print(ankle.head())

# Compute stance vs swing
### Stance = velocity approximately 0
STANCE_VEL_THRESHOLD = 0.02

ankle["L_stance"] = ankle["L_vel"].abs() < STANCE_VEL_THRESHOLD
ankle["R_stance"] = ankle["R_vel"].abs() < STANCE_VEL_THRESHOLD

L_stance_ratio = ankle["L_stance"].mean()
R_stance_ratio = ankle["R_stance"].mean()

print("\nStance ratio:")
print(" Left stance %:", L_stance_ratio * 100)
print(" Right stance %:", R_stance_ratio * 100)

# Compute ankle stiffness
### stiffness = torque / angle

ankle["L_stiffness"] = ankle["L_tau"] / (ankle["L_angle"] + 1e-6)
ankle["R_stiffness"] = ankle["R_tau"] / (ankle["R_angle"] + 1e-6)

# Detect instability spikes
### Torque spikes = sudden instability events

L_spike_threshold = ankle["L_tau"].std() * 3
R_spike_threshold = ankle["R_tau"].std() * 3

ankle["L_spike"] = ankle["L_tau"].abs() > L_spike_threshold
ankle["R_spike"] = ankle["R_tau"].abs() > R_spike_threshold

print("\nInstability spike counts:")
print(" Left ankle spikes:", ankle["L_spike"].sum())
print(" Right ankle spikes:", ankle["R_spike"].sum())

# Stability index:
### Variance of ankle velocity during stance
### Lower variance = more stable

L_stability_index = ankle[ankle["L_stance"]]["L_vel"].var()
R_stability_index = ankle[ankle["R_stance"]]["R_vel"].var()

print("\nStability Index:")
print(" Left ankle:", L_stability_index)
print(" Right ankle:", R_stability_index)

# Plot joint angle trajectories

plt.figure(figsize=(10,4))
plt.plot(ankle["t"], ankle["L_angle"], label="Left Ankle")
plt.plot(ankle["t"], ankle["R_angle"], label="Right Ankle")
plt.title("Ankle Joint Angles")
plt.xlabel("Time")
plt.ylabel("Angle (rad)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/sairam/NU/Assistive/Results/plots/ankle_angles_2.png")
plt.show()

# Plot torques

plt.figure(figsize=(10,4))
plt.plot(ankle["t"], ankle["L_tau"], label="Left Ankle Torque")
plt.plot(ankle["t"], ankle["R_tau"], label="Right Ankle Torque")
plt.title("Ankle Torques")
plt.xlabel("Time")
plt.ylabel("Torque (Nm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/sairam/NU/Assistive/Results/plots/ankle_torques_2.png")
plt.show()

# Phase portrait (angle vs velocity)

plt.figure(figsize=(6,6))
plt.plot(ankle["L_angle"], ankle["L_vel"], ".", alpha=0.7, label="Left")
plt.plot(ankle["R_angle"], ankle["R_vel"], ".", alpha=0.7, label="Right")
plt.title("Ankle Phase Portrait (Angle vs Velocity)")
plt.xlabel("Angle (rad)")
plt.ylabel("Velocity (rad/s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("/home/sairam/NU/Assistive/Results/plots/ankle_phase_portrait_2.png")
plt.show()

# Save processed ankle dataset

ankle.to_csv("/home/sairam/NU/Assistive/Results/csv_files/h1_ankle_processed_2.csv", index=False)
print("\nSaved processed ankle data → h1_ankle_processed.csv")

print("\nAll analysis complete! Plots and CSVs generated.")
