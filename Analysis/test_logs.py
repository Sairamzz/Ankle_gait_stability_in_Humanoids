import numpy as np
import pandas as pd

torques = np.load("/home/sairam/NU/Assistive/Results/npy_files/h1_torque_log_2.npy")
qpos    = np.load("/home/sairam/NU/Assistive/Results/npy_files/h1_qpos_log_2.npy")
qvel    = np.load("/home/sairam/NU/Assistive/Results/npy_files/h1_qvel_log_2.npy")

print("Shapes:", torques.shape, qpos.shape, qvel.shape)

torque_labels = [
"left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle",
"right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle",
"torso",
"left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow",
"right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow"
]

qpos_labels = [
"pelvis_x","pelvis_y","pelvis_z",
"pelvis_qw","pelvis_qx","pelvis_qy","pelvis_qz",
"left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle",
"right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle",
"torso",
"left_shoulder_pitch","left_shoulder_roll","left_shoulder_yaw","left_elbow",
"right_shoulder_pitch","right_shoulder_roll","right_shoulder_yaw","right_elbow"
]

qvel_labels = [
"pelvis_vx","pelvis_vy","pelvis_vz",
"pelvis_wx","pelvis_wy","pelvis_wz",
"left_hip_yaw_vel","left_hip_roll_vel","left_hip_pitch_vel",
"left_knee_vel","left_ankle_vel",
"right_hip_yaw_vel","right_hip_roll_vel","right_hip_pitch_vel",
"right_knee_vel","right_ankle_vel",
"torso_vel",
"left_shoulder_pitch_vel","left_shoulder_roll_vel","left_shoulder_yaw_vel",
"left_elbow_vel",
"right_shoulder_pitch_vel","right_shoulder_roll_vel","right_shoulder_yaw_vel",
"right_elbow_vel"
]

df_torques = pd.DataFrame(torques, columns=torque_labels)
df_qpos    = pd.DataFrame(qpos,    columns=qpos_labels)
df_qvel    = pd.DataFrame(qvel,    columns=qvel_labels)

# Add a time index (optional)
df_torques["time_step"] = np.arange(len(torques))
df_qpos["time_step"]    = np.arange(len(qpos))
df_qvel["time_step"]    = np.arange(len(qvel))

df_torques.to_csv("/home/sairam/NU/Assistive/Results/csv_files/h1_torques_labeled_2.csv", index=False)
df_qpos.to_csv("/home/sairam/NU/Assistive/Results/csv_files/h1_qpos_labeled_2.csv", index=False)
df_qvel.to_csv("/home/sairam/NU/Assistive/Results/csv_files/h1_qvel_labeled_2.csv", index=False)

print("Saved labeled CSV files!")
