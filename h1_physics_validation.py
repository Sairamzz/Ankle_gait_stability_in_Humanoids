import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PhysicsValidatorWithTorques:
    """
    Validate if recorded robot data obeys fundamental physics laws
    Now includes torque validation
    """
    
    def __init__(self, qpos_file, qvel_file, torques_file, ankle_file):
        self.qpos = pd.read_csv(qpos_file)
        self.qvel = pd.read_csv(qvel_file)
        self.torques = pd.read_csv(torques_file)
        self.ankle = pd.read_csv(ankle_file)
        self.dt = 0.02  # Assuming 50Hz
        
        print(f"Loaded data: {len(self.qpos)} samples ({len(self.qpos) * self.dt:.2f}s)")
    
    def check_torque_actuator_limits(self):
        """
        Check if torques are within H1 actuator specifications
        """
        # H1 actuator limits (approximate from specs)
        actuator_limits = {
            'hip_yaw': 45,      # Nm
            'hip_roll': 120,    # Nm
            'hip_pitch': 120,   # Nm
            'knee': 300,        # Nm (high torque joint)
            'ankle': 45,        # Nm
            'torso': 100,       # Nm
            'shoulder_pitch': 30,
            'shoulder_roll': 30,
            'shoulder_yaw': 15,
            'elbow': 30
        }
        
        validation = {}
        violations_count = 0
        
        for joint_type, limit in actuator_limits.items():
            for side in ['left', 'right'] if joint_type != 'torso' else ['']:
                joint_name = f'{side}_{joint_type}' if side else joint_type
                
                if joint_name in self.torques.columns:
                    torques = self.torques[joint_name].values
                    peak_torque = np.max(np.abs(torques))
                    utilization = (peak_torque / limit) * 100
                    
                    passes = peak_torque <= limit
                    if not passes:
                        violations_count += 1
                    
                    validation[joint_name] = {
                        'peak_torque': peak_torque,
                        'limit': limit,
                        'utilization': utilization,
                        'passes': passes,
                        'margin': limit - peak_torque
                    }
        
        return validation, violations_count
    
    def check_power_balance(self):
        """
        Validate Power = Torque × Velocity for all joints
        """
        results = {}
        
        for joint in ['left_ankle', 'right_ankle', 'left_knee', 'right_knee']:
            torque = self.torques[joint].values
            velocity = self.qvel[f'{joint}_vel'].values
            
            # Compute power
            power = torque * velocity
            
            # Check if velocity derivative matches acceleration expectations
            angle_col = joint.replace('_', '_')  # Get corresponding position column
            if joint in self.qpos.columns:
                angle = self.qpos[joint].values
                angle_derivative = np.gradient(angle, self.dt)
                velocity_error = np.mean(np.abs(velocity - angle_derivative))
                
                results[joint] = {
                    'mean_power': np.mean(power),
                    'power_positive_ratio': np.sum(power > 0) / len(power) * 100,
                    'power_negative_ratio': np.sum(power < 0) / len(power) * 100,
                    'velocity_derivative_error': velocity_error,
                    'passes_derivative': velocity_error < 0.1
                }
        
        return results
    
    def check_torque_smoothness(self, max_jerk_threshold=5000):
        """
        Check if torques change smoothly (no unrealistic spikes)
        Jerk = d³position/dt³, but we check torque rate of change
        """
        results = {}
        
        leg_joints = ['left_ankle', 'right_ankle', 'left_knee', 'right_knee',
                      'left_hip_pitch', 'right_hip_pitch']
        
        for joint in leg_joints:
            torques = self.torques[joint].values
            
            # Compute torque rate of change (derivative)
            torque_rate = np.gradient(torques, self.dt)
            
            # Compute second derivative (jerk in torque space)
            torque_jerk = np.gradient(torque_rate, self.dt)
            
            max_jerk = np.max(np.abs(torque_jerk))
            mean_jerk = np.mean(np.abs(torque_jerk))
            
            # Find spikes (>3 std dev)
            jerk_std = np.std(torque_jerk)
            jerk_mean = np.mean(torque_jerk)
            spike_indices = np.where(np.abs(torque_jerk - jerk_mean) > 3 * jerk_std)[0]
            
            results[joint] = {
                'max_jerk': max_jerk,
                'mean_jerk': mean_jerk,
                'num_spikes': len(spike_indices),
                'passes': max_jerk < max_jerk_threshold,
                'spike_indices': spike_indices
            }
        
        return results
    
    def check_torque_grf_consistency(self):
        """
        Validate that ankle torques produce reasonable ground reaction forces
        """
        results = {}
        
        # Typical H1 ankle moment arm ~0.12-0.15m
        moment_arm = 0.13  # meters
        
        for side in ['left', 'right']:
            prefix = 'L' if side == 'left' else 'R'
            
            ankle_torque = self.torques[f'{side}_ankle'].values
            stance_phase = self.ankle[f'{prefix}_stance'].values
            
            # Estimate GRF from ankle torque
            estimated_grf = np.abs(ankle_torque) / moment_arm
            
            # During stance, GRF should be significant (robot weight ~460N)
            stance_grf = estimated_grf[stance_phase == True]
            swing_grf = estimated_grf[stance_phase == False]
            
            if len(stance_grf) > 0:
                mean_stance_grf = np.mean(stance_grf)
                mean_swing_grf = np.mean(swing_grf) if len(swing_grf) > 0 else 0
                
                # Stance GRF should be >> swing GRF
                ratio = mean_stance_grf / (mean_swing_grf + 1e-6)
                
                # Reasonable GRF during stance: 200-800N for 47kg robot
                passes = (mean_stance_grf > 100) and (mean_stance_grf < 1000) and (ratio > 2)
                
                results[side] = {
                    'mean_stance_grf': mean_stance_grf,
                    'mean_swing_grf': mean_swing_grf,
                    'stance_swing_ratio': ratio,
                    'passes': passes
                }
        
        return results
    
    def check_mechanical_power_consistency(self):
        """
        Check if total mechanical power is reasonable
        """
        # Compute total power across all leg joints
        leg_joints = ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
                      'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle']
        
        total_power = np.zeros(len(self.qpos))
        
        for joint in leg_joints:
            torque = self.torques[joint].values
            velocity = self.qvel[f'{joint}_vel'].values
            power = torque * velocity
            total_power += np.abs(power)  # Total absolute power
        
        mean_power = np.mean(total_power)
        peak_power = np.max(total_power)
        
        # For a 47kg humanoid walking at ~1-2 m/s:
        # Mean power: 50-200W is reasonable
        # Peak power: 300-800W is reasonable
        
        passes = (mean_power > 10) and (mean_power < 500) and (peak_power < 1500)
        
        return {
            'mean_power': mean_power,
            'peak_power': peak_power,
            'total_work': np.trapz(total_power, dx=self.dt),
            'passes': passes,
            'power_timeseries': total_power
        }
    
    def check_energy_conservation(self, robot_mass=47.0):
        """
        Check if total energy is roughly conserved (accounting for losses)
        """
        vx = self.qvel['pelvis_vx'].values
        vy = self.qvel['pelvis_vy'].values
        vz = self.qvel['pelvis_vz'].values
        
        v_squared = vx**2 + vy**2 + vz**2
        kinetic_energy = 0.5 * robot_mass * v_squared
        
        height = self.qpos['pelvis_z'].values
        g = 9.81
        potential_energy = robot_mass * g * height
        
        total_energy = kinetic_energy + potential_energy
        
        energy_changes = np.diff(total_energy)
        max_energy_increase = np.max(energy_changes)
        
        validation = {
            'mean_kinetic': np.mean(kinetic_energy),
            'mean_potential': np.mean(potential_energy),
            'total_energy_range': np.max(total_energy) - np.min(total_energy),
            'max_energy_increase_per_step': max_energy_increase,
            'passes': max_energy_increase < 50
        }
        
        return validation, kinetic_energy, potential_energy, total_energy
    
    def visualize_validation_with_torques(self, save_path='physics_validation_with_torques.png'):
        """Create comprehensive validation visualization including torques"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        time = np.arange(len(self.qpos)) * self.dt
        
        # 1. Energy conservation
        energy_val, ke, pe, te = self.check_energy_conservation()
        
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time, ke, 'r-', label='Kinetic', linewidth=2)
        ax1.plot(time, pe, 'b-', label='Potential', linewidth=2)
        ax1.plot(time, te, 'k--', label='Total', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Energy (J)')
        ax1.set_title(f'Energy Conservation (Pass: {energy_val["passes"]})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Torque actuator limits - Ankle
        ax2 = fig.add_subplot(gs[1, 0])
        torque_limits, violations = self.check_torque_actuator_limits()
        
        ax2.plot(time, self.torques['left_ankle'].values, 'b-', label='Left', linewidth=2)
        ax2.plot(time, self.torques['right_ankle'].values, 'r-', label='Right', linewidth=2)
        limit = torque_limits['left_ankle']['limit']
        ax2.axhline(limit, color='orange', linestyle='--', label=f'Limit: ±{limit}Nm')
        ax2.axhline(-limit, color='orange', linestyle='--')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Torque (Nm)')
        ax2.set_title(f'Ankle Torques vs Limits\n(L: {torque_limits["left_ankle"]["utilization"]:.1f}%, R: {torque_limits["right_ankle"]["utilization"]:.1f}%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Torque actuator limits - Knee
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(time, self.torques['left_knee'].values, 'b-', label='Left', linewidth=2)
        ax3.plot(time, self.torques['right_knee'].values, 'r-', label='Right', linewidth=2)
        limit = torque_limits['left_knee']['limit']
        ax3.axhline(limit, color='orange', linestyle='--', label=f'Limit: ±{limit}Nm')
        ax3.axhline(-limit, color='orange', linestyle='--')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Torque (Nm)')
        ax3.set_title(f'Knee Torques vs Limits\n(L: {torque_limits["left_knee"]["utilization"]:.1f}%, R: {torque_limits["right_knee"]["utilization"]:.1f}%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Torque smoothness
        ax4 = fig.add_subplot(gs[1, 2])
        smoothness = self.check_torque_smoothness()
        
        joints_to_plot = ['left_ankle', 'right_ankle']
        colors = ['blue', 'red']
        for joint, color in zip(joints_to_plot, colors):
            torques = self.torques[joint].values
            torque_rate = np.gradient(torques, self.dt)
            ax4.plot(time, torque_rate, color=color, label=joint, linewidth=2)
            
            # Mark spikes
            if smoothness[joint]['num_spikes'] > 0:
                spike_idx = smoothness[joint]['spike_indices']
                ax4.scatter(time[spike_idx], torque_rate[spike_idx], 
                           c=color, s=100, marker='x', zorder=5)
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Torque Rate (Nm/s)')
        ax4.set_title(f'Torque Smoothness\n(Spikes: L={smoothness["left_ankle"]["num_spikes"]}, R={smoothness["right_ankle"]["num_spikes"]})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Power validation
        ax5 = fig.add_subplot(gs[2, 0])
        power_data = self.check_power_balance()
        
        left_power = self.torques['left_ankle'].values * self.qvel['left_ankle_vel'].values
        right_power = self.torques['right_ankle'].values * self.qvel['right_ankle_vel'].values
        
        ax5.plot(time, left_power, 'b-', label='Left', linewidth=2)
        ax5.plot(time, right_power, 'r-', label='Right', linewidth=2)
        ax5.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Power (W)')
        ax5.set_title(f'Ankle Power Balance\n(Pass: {power_data["left_ankle"]["passes_derivative"]} & {power_data["right_ankle"]["passes_derivative"]})')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. GRF consistency
        ax6 = fig.add_subplot(gs[2, 1])
        grf_check = self.check_torque_grf_consistency()
        
        for side, color in [('left', 'blue'), ('right', 'red')]:
            ankle_torque = self.torques[f'{side}_ankle'].values
            estimated_grf = np.abs(ankle_torque) / 0.13
            ax6.plot(time, estimated_grf, color=color, label=side, linewidth=2)
        
        ax6.axhline(460, color='green', linestyle='--', alpha=0.5, label='Robot weight')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Estimated GRF (N)')
        ax6.set_title(f'Ground Reaction Force from Torques\n(Pass: {grf_check["left"]["passes"]} & {grf_check["right"]["passes"]})')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Total mechanical power
        ax7 = fig.add_subplot(gs[2, 2])
        mech_power = self.check_mechanical_power_consistency()
        
        ax7.plot(time, mech_power['power_timeseries'], 'k-', linewidth=2)
        ax7.fill_between(time, 0, mech_power['power_timeseries'], alpha=0.3)
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Total Power (W)')
        ax7.set_title(f'Total Mechanical Power\n(Mean: {mech_power["mean_power"]:.1f}W, Pass: {mech_power["passes"]})')
        ax7.grid(True, alpha=0.3)
        
        # 8-10. Actuator utilization bars
        ax8 = fig.add_subplot(gs[3, :])
        
        joints_to_show = ['left_ankle', 'right_ankle', 'left_knee', 'right_knee',
                          'left_hip_pitch', 'right_hip_pitch']
        utilizations = [torque_limits[j]['utilization'] for j in joints_to_show]
        colors_bar = ['blue' if 'left' in j else 'red' for j in joints_to_show]
        
        bars = ax8.barh(joints_to_show, utilizations, color=colors_bar, alpha=0.7, edgecolor='black')
        ax8.axvline(100, color='orange', linestyle='--', linewidth=2, label='100% (Limit)')
        ax8.axvline(80, color='yellow', linestyle='--', linewidth=1, alpha=0.5, label='80% (Warning)')
        ax8.set_xlabel('Actuator Utilization (%)')
        ax8.set_title('Peak Torque vs Actuator Limits')
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for bar, util in zip(bars, utilizations):
            ax8.text(util + 2, bar.get_y() + bar.get_height()/2, 
                    f'{util:.1f}%', va='center', fontsize=9)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved physics validation to {save_path}")
        plt.show()
    
    def generate_validation_report(self):
        """Generate comprehensive validation report with torques"""
        report = "="*70 + "\n"
        report += "PHYSICS VALIDATION REPORT (WITH TORQUES)\n"
        report += "="*70 + "\n\n"
        
        # Energy conservation
        energy_val, _, _, _ = self.check_energy_conservation()
        report += "1. ENERGY CONSERVATION:\n"
        report += "-"*70 + "\n"
        report += f"   Status: {'PASS ✓' if energy_val['passes'] else 'FAIL ✗'}\n"
        report += f"   Mean Kinetic Energy:     {energy_val['mean_kinetic']:8.2f} J\n"
        report += f"   Mean Potential Energy:   {energy_val['mean_potential']:8.2f} J\n"
        report += f"   Total Energy Range:      {energy_val['total_energy_range']:8.2f} J\n"
        report += f"   Max Energy Increase:     {energy_val['max_energy_increase_per_step']:8.2f} J/step\n\n"
        
        # Torque limits
        torque_limits, violations = self.check_torque_actuator_limits()
        report += "2. TORQUE ACTUATOR LIMITS:\n"
        report += "-"*70 + "\n"
        report += f"   Total Violations: {violations}\n\n"
        
        for joint in ['left_ankle', 'right_ankle', 'left_knee', 'right_knee']:
            data = torque_limits[joint]
            report += f"   {joint:20s}: {data['peak_torque']:8.2f} Nm / {data['limit']:8.2f} Nm "
            report += f"({data['utilization']:5.1f}%) "
            report += f"{'✓' if data['passes'] else '✗ VIOLATION'}\n"
        report += "\n"
        
        # Torque smoothness
        smoothness = self.check_torque_smoothness()
        report += "3. TORQUE SMOOTHNESS:\n"
        report += "-"*70 + "\n"
        for joint in ['left_ankle', 'right_ankle', 'left_knee', 'right_knee']:
            data = smoothness[joint]
            report += f"   {joint:20s}: {data['num_spikes']} spikes, "
            report += f"max jerk={data['max_jerk']:8.1f} Nm/s² "
            report += f"{'✓' if data['passes'] else '✗'}\n"
        report += "\n"
        
        # GRF consistency
        grf_check = self.check_torque_grf_consistency()
        report += "4. TORQUE-GRF CONSISTENCY:\n"
        report += "-"*70 + "\n"
        for side in ['left', 'right']:
            data = grf_check[side]
            report += f"   {side.upper()}:\n"
            report += f"     Mean Stance GRF:    {data['mean_stance_grf']:8.2f} N\n"
            report += f"     Mean Swing GRF:     {data['mean_swing_grf']:8.2f} N\n"
            report += f"     Ratio:              {data['stance_swing_ratio']:8.2f}\n"
            report += f"     Status:             {'✓' if data['passes'] else '✗'}\n\n"
        
        # Mechanical power
        mech_power = self.check_mechanical_power_consistency()
        report += "5. MECHANICAL POWER:\n"
        report += "-"*70 + "\n"
        report += f"   Status: {'PASS ✓' if mech_power['passes'] else 'FAIL ✗'}\n"
        report += f"   Mean Power:     {mech_power['mean_power']:8.2f} W\n"
        report += f"   Peak Power:     {mech_power['peak_power']:8.2f} W\n"
        report += f"   Total Work:     {mech_power['total_work']:8.2f} J\n\n"
        
        report += "="*70 + "\n"
        
        return report

# Usage
if __name__ == "__main__":
    validator = PhysicsValidatorWithTorques(
        'h1_qpos_labeled_2.csv',
        'h1_qvel_labeled_2.csv',
        'h1_torques_labeled_2.csv',
        'h1_ankle_processed_2.csv'
    )
    
    # Generate report
    report = validator.generate_validation_report()
    print(report)
    
    # Save report
    with open('physics_validation_with_torques_report.txt', 'w') as f:
        f.write(report)
    
    # Create visualizations
    validator.visualize_validation_with_torques('physics_validation_with_torques.png')
    
    print("\n✅ Validation complete!")
