import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_simulation_with_load(load_mass_kg, config, save_dir):
    """Run simulation with specified torso load and collect ankle data"""
    
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    
    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    
    num_actions = config["num_actions"]
    num_obs = config["num_obs"]
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    
    # Initialize
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    
    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    
    # ADD MASS TO TORSO
    torso_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
    original_mass = m.body_mass[torso_body_id]
    m.body_mass[torso_body_id] = original_mass + load_mass_kg
    print(f"Added {load_mass_kg} kg to torso (original: {original_mass:.2f} kg, new: {m.body_mass[torso_body_id]:.2f} kg)")
    
    # Find ankle joint indices
    # G1 has 2 ankle joints per leg: pitch (forward/back) and roll (side-to-side)
    # We'll track both but focus on pitch for gait analysis
    left_ankle_pitch_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'left_ankle_pitch_joint')
    right_ankle_pitch_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_ankle_pitch_joint')
    left_ankle_roll_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'left_ankle_roll_joint')
    right_ankle_roll_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_ankle_roll_joint')
    
    print(f"Ankle joint indices - Left pitch: {left_ankle_pitch_id}, Right pitch: {right_ankle_pitch_id}")
    print(f"                      Left roll: {left_ankle_roll_id}, Right roll: {right_ankle_roll_id}")
    
    # Data collection arrays
    data_log = {
        'time': [],
        'left_ankle_pitch_pos': [],
        'right_ankle_pitch_pos': [],
        'left_ankle_pitch_vel': [],
        'right_ankle_pitch_vel': [],
        'left_ankle_pitch_torque': [],
        'right_ankle_pitch_torque': [],
        'left_ankle_roll_pos': [],
        'right_ankle_roll_pos': [],
        'left_ankle_roll_vel': [],
        'right_ankle_roll_vel': [],
        'left_ankle_roll_torque': [],
        'right_ankle_roll_torque': [],
        'com_x': [],
        'com_y': [],
        'com_z': [],
        'base_height': [],
        'base_roll': [],
        'base_pitch': [],
        'base_yaw': [],
    }
    
    # Load policy
    policy = torch.jit.load(policy_path)
    
    print(f"Running simulation with {load_mass_kg} kg load for {simulation_duration} seconds...")
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # PD control
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            
            # Step physics
            mujoco.mj_step(m, d)
            counter += 1
            
            # Log data every 10 steps (50Hz logging rate)
            if counter % 10 == 0:
                data_log['time'].append(d.time)
                
                # Get joint indices in qpos/qvel arrays (subtract 7/6 for base offset)
                left_pitch_qpos_idx = m.jnt_qposadr[left_ankle_pitch_id] - 7
                right_pitch_qpos_idx = m.jnt_qposadr[right_ankle_pitch_id] - 7
                left_roll_qpos_idx = m.jnt_qposadr[left_ankle_roll_id] - 7
                right_roll_qpos_idx = m.jnt_qposadr[right_ankle_roll_id] - 7
                
                left_pitch_qvel_idx = m.jnt_dofadr[left_ankle_pitch_id] - 6
                right_pitch_qvel_idx = m.jnt_dofadr[right_ankle_pitch_id] - 6
                left_roll_qvel_idx = m.jnt_dofadr[left_ankle_roll_id] - 6
                right_roll_qvel_idx = m.jnt_dofadr[right_ankle_roll_id] - 6
                
                # Pitch joints
                data_log['left_ankle_pitch_pos'].append(d.qpos[7 + left_pitch_qpos_idx])
                data_log['right_ankle_pitch_pos'].append(d.qpos[7 + right_pitch_qpos_idx])
                data_log['left_ankle_pitch_vel'].append(d.qvel[6 + left_pitch_qvel_idx])
                data_log['right_ankle_pitch_vel'].append(d.qvel[6 + right_pitch_qvel_idx])
                data_log['left_ankle_pitch_torque'].append(tau[left_pitch_qvel_idx])
                data_log['right_ankle_pitch_torque'].append(tau[right_pitch_qvel_idx])
                
                # Roll joints
                data_log['left_ankle_roll_pos'].append(d.qpos[7 + left_roll_qpos_idx])
                data_log['right_ankle_roll_pos'].append(d.qpos[7 + right_roll_qpos_idx])
                data_log['left_ankle_roll_vel'].append(d.qvel[6 + left_roll_qvel_idx])
                data_log['right_ankle_roll_vel'].append(d.qvel[6 + right_roll_qvel_idx])
                data_log['left_ankle_roll_torque'].append(tau[left_roll_qvel_idx])
                data_log['right_ankle_roll_torque'].append(tau[right_roll_qvel_idx])
                
                # COM position
                data_log['com_x'].append(d.subtree_com[0][0])
                data_log['com_y'].append(d.subtree_com[0][1])
                data_log['com_z'].append(d.subtree_com[0][2])
                
                # Base pose
                data_log['base_height'].append(d.qpos[2])
                # Convert quaternion to euler for easier analysis
                quat = d.qpos[3:7]
                gravity_vec = get_gravity_orientation(quat)
                data_log['base_roll'].append(np.arctan2(gravity_vec[1], gravity_vec[2]))
                data_log['base_pitch'].append(np.arctan2(-gravity_vec[0], np.sqrt(gravity_vec[1]**2 + gravity_vec[2]**2)))
                data_log['base_yaw'].append(0.0)  # Simplified
            
            # Policy control
            if counter % control_decimation == 0:
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]
                
                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale
                
                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)
                
                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * action_scale + default_angles
            
            viewer.sync()
            
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    
    # Save data
    df = pd.DataFrame(data_log)
    csv_path = save_dir / f'ankle_data_{load_mass_kg}kg.csv'
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    
    return df


def analyze_and_plot(data_dict, save_dir):
    """Generate analysis plots from collected data"""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Ankle Load Analysis - G1 Humanoid', fontsize=16)
    
    # Plot 1: Ankle Pitch Torque Time Series
    ax = axes[0, 0]
    for load, df in data_dict.items():
        ax.plot(df['time'], df['left_ankle_pitch_torque'], label=f'{load}kg - Left', alpha=0.7)
        ax.plot(df['time'], df['right_ankle_pitch_torque'], label=f'{load}kg - Right', alpha=0.7, linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Ankle Pitch Torque (Nm)')
    ax.set_title('Ankle Pitch Torque Over Time')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Peak Torque vs Load (Both Pitch and Roll)
    ax = axes[0, 1]
    loads = []
    left_pitch_peak = []
    right_pitch_peak = []
    left_roll_peak = []
    right_roll_peak = []
    for load, df in data_dict.items():
        loads.append(load)
        left_pitch_peak.append(df['left_ankle_pitch_torque'].abs().max())
        right_pitch_peak.append(df['right_ankle_pitch_torque'].abs().max())
        left_roll_peak.append(df['left_ankle_roll_torque'].abs().max())
        right_roll_peak.append(df['right_ankle_roll_torque'].abs().max())
    ax.plot(loads, left_pitch_peak, 'o-', label='Left Pitch', markersize=10, linewidth=2)
    ax.plot(loads, right_pitch_peak, 's-', label='Right Pitch', markersize=10, linewidth=2)
    ax.plot(loads, left_roll_peak, '^--', label='Left Roll', markersize=8, alpha=0.6)
    ax.plot(loads, right_roll_peak, 'v--', label='Right Roll', markersize=8, alpha=0.6)
    ax.set_xlabel('Torso Load (kg)')
    ax.set_ylabel('Peak Ankle Torque (Nm)')
    ax.set_title('Peak Ankle Torque vs Load')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: COM Sway
    ax = axes[1, 0]
    for load, df in data_dict.items():
        ax.plot(df['com_x'], df['com_y'], label=f'{load}kg', alpha=0.6)
    ax.set_xlabel('COM X (m)')
    ax.set_ylabel('COM Y (m)')
    ax.set_title('Center of Mass Trajectory (Top View)')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Plot 4: Base Height Stability
    ax = axes[1, 1]
    for load, df in data_dict.items():
        ax.plot(df['time'], df['base_height'], label=f'{load}kg', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Base Height (m)')
    ax.set_title('Base Height Stability')
    ax.legend()
    ax.grid(True)
    
    # Plot 5: Ankle Position (Gait Pattern) - Pitch joints
    ax = axes[2, 0]
    for load, df in data_dict.items():
        # Only plot first 10 seconds for clarity
        mask = df['time'] <= 10.0
        ax.plot(df[mask]['time'], df[mask]['left_ankle_pitch_pos'], label=f'{load}kg - Left', alpha=0.7)
        ax.plot(df[mask]['time'], df[mask]['right_ankle_pitch_pos'], label=f'{load}kg - Right', alpha=0.7, linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Ankle Pitch Position (rad)')
    ax.set_title('Ankle Pitch Joint Positions (First 10s)')
    ax.legend()
    ax.grid(True)
    
    # Plot 6: Summary Statistics
    ax = axes[2, 1]
    loads = []
    com_sway = []
    height_std = []
    for load, df in data_dict.items():
        loads.append(load)
        # COM sway = std of lateral position
        com_sway.append(df['com_y'].std())
        height_std.append(df['base_height'].std())
    
    ax2 = ax.twinx()
    p1 = ax.plot(loads, com_sway, 'o-', color='tab:blue', label='COM Lateral Sway', markersize=10)
    p2 = ax2.plot(loads, height_std, 's-', color='tab:orange', label='Height Variation', markersize=10)
    
    ax.set_xlabel('Torso Load (kg)')
    ax.set_ylabel('COM Lateral Sway (m)', color='tab:blue')
    ax2.set_ylabel('Height Std Dev (m)', color='tab:orange')
    ax.set_title('Stability Metrics vs Load')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax.grid(True)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plot_path = save_dir / 'ankle_analysis_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {plot_path}")
    plt.show()
    
    # Generate summary statistics
    summary_path = save_dir / 'analysis_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("ANKLE LOAD ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        for load, df in data_dict.items():
            f.write(f"Load: {load} kg\n")
            f.write(f"  Left Ankle Pitch - Peak Torque:  {df['left_ankle_pitch_torque'].abs().max():.2f} Nm\n")
            f.write(f"  Right Ankle Pitch - Peak Torque: {df['right_ankle_pitch_torque'].abs().max():.2f} Nm\n")
            f.write(f"  Left Ankle Pitch - RMS Torque:   {np.sqrt((df['left_ankle_pitch_torque']**2).mean()):.2f} Nm\n")
            f.write(f"  Right Ankle Pitch - RMS Torque:  {np.sqrt((df['right_ankle_pitch_torque']**2).mean()):.2f} Nm\n")
            f.write(f"  Left Ankle Roll - Peak Torque:   {df['left_ankle_roll_torque'].abs().max():.2f} Nm\n")
            f.write(f"  Right Ankle Roll - Peak Torque:  {df['right_ankle_roll_torque'].abs().max():.2f} Nm\n")
            f.write(f"  COM Lateral Sway (std):          {df['com_y'].std():.4f} m\n")
            f.write(f"  Height Variation (std):          {df['base_height'].std():.4f} m\n")
            f.write(f"  Base Roll Variation:             {df['base_roll'].std():.4f} rad\n")
            f.write("\n")
    
    print(f"Summary statistics saved to {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    parser.add_argument("--loads", type=float, nargs='+', default=[0, 5, 10, 15],
                        help="List of torso loads to test (kg)")
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Simulation duration per load (seconds)")
    args = parser.parse_args()
    
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Override simulation duration
    config["simulation_duration"] = args.duration
    
    # Create output directory
    save_dir = Path(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/ankle_analysis_results")
    save_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("G1 ANKLE LOAD ANALYSIS")
    print("="*60)
    print(f"Testing loads: {args.loads} kg")
    print(f"Duration per test: {args.duration} seconds")
    print(f"Results will be saved to: {save_dir}")
    print("="*60 + "\n")
    
    # Run simulations for each load
    data_dict = {}
    for load in args.loads:
        df = run_simulation_with_load(load, config, save_dir)
        data_dict[load] = df
        print()
    
    # Generate analysis plots
    print("Generating analysis plots...")
    analyze_and_plot(data_dict, save_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print(f"Check {save_dir} for results")
    print("="*60)
