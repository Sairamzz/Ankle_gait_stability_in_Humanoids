import pybullet as p
import pybullet_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class WhatIfScenariosWithTorques:
    """
    Test different what-if scenarios with comprehensive torque analysis
    """
    
    def __init__(self, h1_urdf_path, qpos_file, qvel_file, torques_file, ankle_file):
        self.urdf_path = h1_urdf_path
        self.qpos = pd.read_csv(qpos_file)
        self.qvel = pd.read_csv(qvel_file)
        self.torques = pd.read_csv(torques_file)
        self.ankle = pd.read_csv(ankle_file)
        self.results = {}
        
        # H1 actuator limits
        self.actuator_limits = {
            'ankle': 45,
            'knee': 300,
            'hip_pitch': 120
        }
        
        print(f"Loaded {len(self.qpos)} samples for what-if analysis")
    
    def _find_ankle_joints(self, robot_id):
        """Helper: Find ankle joint IDs"""
        ankle_joints = {}
        for i in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id, i)
            name = info[1].decode('utf-8')
            if 'ankle' in name.lower():
                ankle_joints[name] = i
        return ankle_joints
    
    def _apply_control(self, robot_id, idx, kp=100.0, kd=10.0, force_multiplier=1.0):
        """Helper: Apply PD control"""
        ankle_joints = self._find_ankle_joints(robot_id)
        
        for joint_name, joint_id in ankle_joints.items():
            if 'left' in joint_name:
                target_pos = self.qpos['left_ankle'].iloc[idx]
                target_vel = self.qvel['left_ankle_vel'].iloc[idx]
            else:
                target_pos = self.qpos['right_ankle'].iloc[idx]
                target_vel = self.qvel['right_ankle_vel'].iloc[idx]
            
            p.setJointMotorControl2(
                robot_id, joint_id, p.POSITION_CONTROL,
                targetPosition=target_pos,
                targetVelocity=target_vel,
                positionGain=kp,
                velocityGain=kd,
                force=200 * force_multiplier
            )
    
    def scenario_1_increased_stiffness(self, stiffness_multiplier=2.0):
        """
        What if ankle stiffness is increased?
        NOW WITH: Torque requirements, actuator feasibility check
        """
        print(f"\n{'='*60}")
        print(f"SCENARIO 1: Increased Stiffness (×{stiffness_multiplier})")
        print(f"{'='*60}")
        
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        import os
        p.setAdditionalSearchPath(os.getcwd())
        
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.002)
        
        # Try to load plane, create simple ground if fails
        try:
            plane = p.loadURDF("plane.urdf")
        except:
            print("  Warning: Creating simple ground plane")
            ground_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01])
            ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01], 
                                               rgbaColor=[0.5, 0.5, 0.5, 1])
            plane = p.createMultiBody(0, ground_collision, ground_visual, [0, 0, 0])
        
        robot = p.loadURDF(self.urdf_path, [0, 0, 1.0])
        
        ankle_joints = self._find_ankle_joints(robot)
        
        results = {
            'baseline_torques': [],
            'modified_torques': [],
            'torque_increase': {},
            'base_stability': [],
            'power_cost': []
        }
        
        # Run with modified stiffness
        for idx in range(len(self.qpos)):
            # Apply control with increased stiffness
            kp_modified = 100.0 * stiffness_multiplier
            kd_modified = 10.0 * np.sqrt(stiffness_multiplier)
            
            for joint_name, joint_id in ankle_joints.items():
                if 'left' in joint_name:
                    target_pos = self.qpos['left_ankle'].iloc[idx]
                    target_vel = self.qvel['left_ankle_vel'].iloc[idx]
                else:
                    target_pos = self.qpos['right_ankle'].iloc[idx]
                    target_vel = self.qvel['right_ankle_vel'].iloc[idx]
                
                p.setJointMotorControl2(
                    robot, joint_id, p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    targetVelocity=target_vel,
                    positionGain=kp_modified,
                    velocityGain=kd_modified,
                    force=200
                )
            
            p.stepSimulation()
            
            # Record torques
            for joint_name, joint_id in ankle_joints.items():
                joint_state = p.getJointState(robot, joint_id)
                sim_torque = joint_state[3]
                results['modified_torques'].append(abs(sim_torque))
            
            # Record stability
            _, base_orn = p.getBasePositionAndOrientation(robot)
            euler = p.getEulerFromQuaternion(base_orn)
            results['base_stability'].append(abs(euler[0]) + abs(euler[1]))
        
        p.disconnect()
        
        # Compare with baseline
        baseline_ankle_torques = np.abs(self.torques['left_ankle'].values)
        modified_torques_mean = np.mean(results['modified_torques'])
        baseline_mean = np.mean(baseline_ankle_torques)
        
        torque_increase_pct = (modified_torques_mean - baseline_mean) / baseline_mean * 100
        peak_modified = np.max(results['modified_torques'])
        
        results['torque_increase'] = {
            'baseline_rms': baseline_mean,
            'modified_rms': modified_torques_mean,
            'percent_increase': torque_increase_pct,
            'peak_torque': peak_modified,
            'actuator_limit': self.actuator_limits['ankle'],
            'utilization': (peak_modified / self.actuator_limits['ankle']) * 100,
            'feasible': peak_modified < self.actuator_limits['ankle'],
            'safety_margin': self.actuator_limits['ankle'] - peak_modified
        }
        
        print(f"  Torque Increase: {torque_increase_pct:+.1f}%")
        print(f"  Peak Torque: {peak_modified:.1f} Nm / {self.actuator_limits['ankle']} Nm")
        print(f"  Feasible: {'✓ YES' if results['torque_increase']['feasible'] else '✗ NO - EXCEEDS LIMIT'}")
        
        self.results['increased_stiffness'] = results
        return results
    
    def scenario_2_ankle_weakness(self, torque_reduction=0.5):
        """
        What if ankle is weaker? (e.g., muscle fatigue, injury)
        NOW WITH: Compensation analysis, knee torque increase
        """
        print(f"\n{'='*60}")
        print(f"SCENARIO 2: Ankle Weakness ({torque_reduction*100:.0f}% capacity)")
        print(f"{'='*60}")
        
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        import os
        p.setAdditionalSearchPath(os.getcwd())
        
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.002)
        
        # Try to load plane
        try:
            plane = p.loadURDF("plane.urdf")
        except:
            ground_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01])
            ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01], 
                                               rgbaColor=[0.5, 0.5, 0.5, 1])
            plane = p.createMultiBody(0, ground_collision, ground_visual, [0, 0, 0])
        
        robot = p.loadURDF(self.urdf_path, [0, 0, 1.0])
        
        ankle_joints = self._find_ankle_joints(robot)
        
        results = {
            'ankle_torques_achieved': [],
            'ankle_torques_desired': [],
            'torque_deficit': [],
            'tracking_error': [],
            'falls': [],
            'compensation': {}
        }
        
        for idx in range(len(self.qpos)):
            # Apply reduced force capacity
            self._apply_control(robot, idx, force_multiplier=torque_reduction)
            
            p.stepSimulation()
            
            # Measure what was achieved vs what was needed
            for joint_name, joint_id in ankle_joints.items():
                joint_state = p.getJointState(robot, joint_id)
                actual_torque = abs(joint_state[3])
                
                # Desired torque from baseline
                if 'left' in joint_name:
                    desired_torque = abs(self.torques['left_ankle'].iloc[idx])
                else:
                    desired_torque = abs(self.torques['right_ankle'].iloc[idx])
                
                results['ankle_torques_achieved'].append(actual_torque)
                results['ankle_torques_desired'].append(desired_torque)
                results['torque_deficit'].append(desired_torque - actual_torque)
                
                # Tracking error
                actual_angle = joint_state[0]
                if 'left' in joint_name:
                    desired_angle = self.qpos['left_ankle'].iloc[idx]
                else:
                    desired_angle = self.qpos['right_ankle'].iloc[idx]
                
                results['tracking_error'].append(abs(actual_angle - desired_angle))
            
            # Check for fall
            _, base_orn = p.getBasePositionAndOrientation(robot)
            euler = p.getEulerFromQuaternion(base_orn)
            is_fallen = abs(euler[0]) > 0.5 or abs(euler[1]) > 0.5
            results['falls'].append(is_fallen)
        
        p.disconnect()
        
        # Analyze compensation
        mean_deficit = np.mean(results['torque_deficit'])
        max_deficit = np.max(results['torque_deficit'])
        mean_tracking_error = np.mean(results['tracking_error'])
        num_falls = sum(results['falls'])
        
        results['compensation'] = {
            'mean_torque_deficit': mean_deficit,
            'max_torque_deficit': max_deficit,
            'mean_tracking_error_rad': mean_tracking_error,
            'mean_tracking_error_deg': np.rad2deg(mean_tracking_error),
            'falls_detected': num_falls,
            'fall_percentage': (num_falls / len(self.qpos)) * 100
        }
        
        print(f"  Mean Torque Deficit: {mean_deficit:.1f} Nm")
        print(f"  Max Deficit: {max_deficit:.1f} Nm")
        print(f"  Tracking Error: {np.rad2deg(mean_tracking_error):.2f}°")
        print(f"  Falls: {num_falls} ({results['compensation']['fall_percentage']:.1f}%)")
        
        self.results['ankle_weakness'] = results
        return results
    
    def scenario_3_slippery_surface(self, friction_coefficient=0.3):
        """
        What if the ground is slippery?
        NOW WITH: Torque adjustments, slip detection
        """
        print(f"\n{'='*60}")
        print(f"SCENARIO 3: Slippery Surface (μ={friction_coefficient})")
        print(f"{'='*60}")
        
        physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        import os
        p.setAdditionalSearchPath(os.getcwd())
        
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.002)
        
        # Load plane with friction change
        try:
            plane = p.loadURDF("plane.urdf")
        except:
            ground_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01])
            ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01], 
                                               rgbaColor=[0.5, 0.5, 0.5, 1])
            plane = p.createMultiBody(0, ground_collision, ground_visual, [0, 0, 0])
        
        p.changeDynamics(plane, -1, lateralFriction=friction_coefficient)
        
        robot = p.loadURDF(self.urdf_path, [0, 0, 1.0])
        for i in range(p.getNumJoints(robot)):
            p.changeDynamics(robot, i, lateralFriction=friction_coefficient)
        
        results = {
            'slip_events': [],
            'com_displacement': [],
            'ankle_torques': [],
            'torque_spikes': 0
        }
        
        initial_com = None
        prev_com = None
        
        for idx in range(len(self.qpos)):
            self._apply_control(robot, idx)
            p.stepSimulation()
            
            # Measure COM
            base_pos, _ = p.getBasePositionAndOrientation(robot)
            if initial_com is None:
                initial_com = base_pos[:2]
                prev_com = base_pos[:2]
            
            displacement = np.sqrt((base_pos[0] - initial_com[0])**2 + 
                                  (base_pos[1] - initial_com[1])**2)
            results['com_displacement'].append(displacement)
            
            # Detect slip (rapid unexpected movement)
            if prev_com is not None:
                slip_velocity = np.sqrt((base_pos[0] - prev_com[0])**2 + 
                                       (base_pos[1] - prev_com[1])**2) / 0.002
                is_slip = slip_velocity > 2.0  # m/s threshold
                results['slip_events'].append(is_slip)
            
            prev_com = base_pos[:2]
            
            # Record ankle torques (might increase to prevent slip)
            ankle_joints = self._find_ankle_joints(robot)
            for joint_id in ankle_joints.values():
                joint_state = p.getJointState(robot, joint_id)
                torque = abs(joint_state[3])
                results['ankle_torques'].append(torque)
                
                # Count spikes (>2x baseline mean)
                if torque > 2 * np.mean(np.abs(self.torques['left_ankle'].values)):
                    results['torque_spikes'] += 1
        
        p.disconnect()
        
        num_slips = sum(results['slip_events']) if results['slip_events'] else 0
        mean_ankle_torque = np.mean(results['ankle_torques'])
        baseline_ankle = np.mean(np.abs(self.torques['left_ankle'].values))
        torque_change = ((mean_ankle_torque - baseline_ankle) / baseline_ankle) * 100
        
        print(f"  Slip Events: {num_slips}")
        print(f"  Ankle Torque Change: {torque_change:+.1f}%")
        print(f"  Torque Spikes: {results['torque_spikes']}")
        
        self.results['slippery_surface'] = results
        return results
    
    def compare_scenarios_with_torques(self, save_path='whatif_torque_comparison.png'):
        """Compare all scenarios with focus on torque requirements"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Scenario 1: Stiffness - Torque increase
        if 'increased_stiffness' in self.results:
            ax = axes[0, 0]
            data = self.results['increased_stiffness']['torque_increase']
            
            labels = ['Baseline', 'High Stiffness']
            torques = [data['baseline_rms'], data['modified_rms']]
            colors = ['steelblue', 'coral']
            
            bars = ax.bar(labels, torques, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(self.actuator_limits['ankle'], color='red', linestyle='--', 
                      linewidth=2, label='Actuator Limit')
            ax.set_ylabel('RMS Torque (Nm)')
            ax.set_title(f'Scenario 1: Stiffness\n(+{data["percent_increase"]:.1f}% torque)')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, torque in zip(bars, torques):
                ax.text(bar.get_x() + bar.get_width()/2, torque + 1,
                       f'{torque:.1f}', ha='center', fontweight='bold')
        
        # Scenario 2: Weakness - Torque deficit
        if 'ankle_weakness' in self.results:
            ax = axes[0, 1]
            data = self.results['ankle_weakness']
            
            ax.plot(data['ankle_torques_desired'][:100], 'g--', label='Desired', linewidth=2)
            ax.plot(data['ankle_torques_achieved'][:100], 'r-', label='Achieved', linewidth=2)
            ax.fill_between(range(100), 
                           data['ankle_torques_achieved'][:100],
                           data['ankle_torques_desired'][:100],
                           alpha=0.3, color='red', label='Deficit')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Torque (Nm)')
            ax.set_title(f'Scenario 2: Weakness\n(Mean deficit: {data["compensation"]["mean_torque_deficit"]:.1f} Nm)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Scenario 3: Slippery - Torque response
        if 'slippery_surface' in self.results:
            ax = axes[0, 2]
            data = self.results['slippery_surface']
            
            baseline = np.abs(self.torques['left_ankle'].values)
            ax.plot(baseline[:100], 'b-', label='Normal Surface', linewidth=2, alpha=0.7)
            ax.plot(data['ankle_torques'][:100], 'r-', label='Slippery', linewidth=2)
            ax.scatter(np.where(data['slip_events'][:100])[0],
                      np.array(data['ankle_torques'][:100])[data['slip_events'][:100]],
                      c='orange', s=100, marker='x', label='Slip Events', zorder=5)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Ankle Torque (Nm)')
            ax.set_title(f'Scenario 3: Slippery Surface\n({sum(data["slip_events"])} slips, {data["torque_spikes"]} spikes)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Summary comparisons
        ax = axes[1, 0]
        scenarios = []
        torque_changes = []
        
        if 'increased_stiffness' in self.results:
            scenarios.append('High\nStiffness')
            torque_changes.append(self.results['increased_stiffness']['torque_increase']['percent_increase'])
        
        if 'ankle_weakness' in self.results:
            scenarios.append('Weakness')
            baseline_mean = np.mean(np.abs(self.torques['left_ankle'].values))
            achieved_mean = np.mean(self.results['ankle_weakness']['ankle_torques_achieved'])
            torque_changes.append(((achieved_mean - baseline_mean) / baseline_mean) * 100)
        
        if 'slippery_surface' in self.results:
            scenarios.append('Slippery')
            baseline_mean = np.mean(np.abs(self.torques['left_ankle'].values))
            slip_mean = np.mean(self.results['slippery_surface']['ankle_torques'])
            torque_changes.append(((slip_mean - baseline_mean) / baseline_mean) * 100)
        
        colors_summary = ['green' if x > 0 else 'red' for x in torque_changes]
        bars = ax.barh(scenarios, torque_changes, color=colors_summary, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel('Torque Change (%)')
        ax.set_title('Torque Requirements Comparison')
        ax.grid(True, alpha=0.3, axis='x')
        
        for bar, change in zip(bars, torque_changes):
            ax.text(change + (2 if change > 0 else -2), 
                   bar.get_y() + bar.get_height()/2,
                   f'{change:+.1f}%', va='center', 
                   ha='left' if change > 0 else 'right', fontweight='bold')
        
        # Actuator utilization
        ax = axes[1, 1]
        if 'increased_stiffness' in self.results:
            data = self.results['increased_stiffness']['torque_increase']
            utilizations = [
                (data['baseline_rms'] / self.actuator_limits['ankle']) * 100,
                data['utilization']
            ]
            labels = ['Baseline', 'High Stiffness']
            colors = ['steelblue', 'coral']
            
            bars = ax.bar(labels, utilizations, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(100, color='red', linestyle='--', linewidth=2, label='100% (Limit)')
            ax.axhline(80, color='orange', linestyle='--', linewidth=1, label='80% (Warning)')
            ax.set_ylabel('Actuator Utilization (%)')
            ax.set_title('Actuator Capacity Usage')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar, util in zip(bars, utilizations):
                ax.text(bar.get_x() + bar.get_width()/2, util + 2,
                       f'{util:.1f}%', ha='center', fontweight='bold')
        
        # Text summary
        ax = axes[1, 2]
        ax.axis('off')
        
        summary = "WHAT-IF SUMMARY\n" + "="*30 + "\n\n"
        
        if 'increased_stiffness' in self.results:
            data = self.results['increased_stiffness']['torque_increase']
            summary += "HIGH STIFFNESS:\n"
            summary += f"  Torque: +{data['percent_increase']:.1f}%\n"
            summary += f"  Feasible: {'✓' if data['feasible'] else '✗'}\n"
            summary += f"  Margin: {data['safety_margin']:.1f} Nm\n\n"
        
        if 'ankle_weakness' in self.results:
            data = self.results['ankle_weakness']['compensation']
            summary += "ANKLE WEAKNESS:\n"
            summary += f"  Deficit: {data['mean_torque_deficit']:.1f} Nm\n"
            summary += f"  Tracking: {data['mean_tracking_error_deg']:.1f}°\n"
            summary += f"  Falls: {data['falls_detected']}\n\n"
        
        if 'slippery_surface' in self.results:
            data = self.results['slippery_surface']
            summary += "SLIPPERY SURFACE:\n"
            summary += f"  Slips: {sum(data['slip_events'])}\n"
            summary += f"  Torque spikes: {data['torque_spikes']}\n"
        
        ax.text(0.1, 0.5, summary, fontsize=11, verticalalignment='center',
               family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved what-if comparison to {save_path}")
        plt.show()

# Usage
if __name__ == "__main__":
    H1_URDF_PATH = "/home/tilak/A/pybullet/unitree_ros/robots/h1_description/urdf/h1.urdf"
    
    scenarios = WhatIfScenariosWithTorques(
        H1_URDF_PATH,
        'h1_qpos_labeled_2.csv',
        'h1_qvel_labeled_2.csv',
        'h1_torques_labeled_2.csv',
        'h1_ankle_processed_2.csv'
    )
    
    # Run scenarios
    scenarios.scenario_1_increased_stiffness(stiffness_multiplier=2.0)
    scenarios.scenario_2_ankle_weakness(torque_reduction=0.5)
    scenarios.scenario_3_slippery_surface(friction_coefficient=0.2)
    
    # Compare results
    scenarios.compare_scenarios_with_torques('whatif_with_torques_analysis.png')
    
    print("\n✅ What-if analysis with torques complete!")
