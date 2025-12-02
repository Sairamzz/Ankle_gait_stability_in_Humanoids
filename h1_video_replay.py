import pybullet as p
import pybullet_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os

class H1TorqueVideoGenerator:
    """
    Generate video frames with robot simulation AND torque data overlays
    """
    
    def __init__(self, h1_urdf_path, frame_dir="h1_torque_frames"):
        self.physics_client = p.connect(p.DIRECT)
        
        self.frame_dir = frame_dir
        self.frame_count = 0
        os.makedirs(self.frame_dir, exist_ok=True)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setAdditionalSearchPath(os.getcwd())
        
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.002)
        
        # Load ground
        try:
            self.plane_id = p.loadURDF("plane.urdf")
        except:
            ground_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01])
            ground_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[10, 10, 0.01], 
                                               rgbaColor=[0.5, 0.5, 0.5, 1])
            self.plane_id = p.createMultiBody(0, ground_collision, ground_visual, [0, 0, 0])
        
        # Load H1 robot
        self.robot_id = p.loadURDF(h1_urdf_path, [0, 0, 1.0], useFixedBase=False)
        
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_info = self._get_joint_info()
        self.joint_name_to_id = {info['name']: jid for jid, info in self.joint_info.items()}
        
        print(f"✅ H1 Robot loaded for torque video generation")
    
    def _get_joint_info(self):
        joint_info = {}
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_info[i] = {
                'name': info[1].decode('utf-8'),
                'type': info[2],
            }
        return joint_info
    
    def load_data(self, qpos_file, qvel_file, torques_file, ankle_file):
        """Load all trajectory and torque data"""
        self.qpos = pd.read_csv(qpos_file)
        self.qvel = pd.read_csv(qvel_file)
        self.torques = pd.read_csv(torques_file)
        self.ankle = pd.read_csv(ankle_file)
        
        self.n_samples = len(self.qpos)
        self.dt = 0.02
        self.time = np.arange(self.n_samples) * self.dt
        
        print(f"📊 Data loaded: {self.n_samples} samples ({self.n_samples * self.dt:.2f} s)")
    
    def capture_robot_view(self, frame_idx):
        """Capture robot from PyBullet camera"""
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        
        # Camera follows robot
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=3.0,
            yaw=45,
            pitch=-20,
            roll=0,
            upAxisIndex=2
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=16/9,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Capture at 1280x720 (HD)
        width, height = 1280, 720
        img = p.getCameraImage(
            width, height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        rgb_array = np.array(img[2]).reshape(height, width, 4)[:, :, :3]
        return rgb_array
    
    def create_torque_overlay(self, frame_idx, width=640, height=720):
        """Create torque graphs overlay"""
        fig = plt.figure(figsize=(6.4, 7.2), dpi=100, facecolor='white')
        gs = fig.add_gridspec(4, 1, hspace=0.4, top=0.95, bottom=0.05, left=0.12, right=0.95)
        
        # Current time marker
        current_time = self.time[frame_idx]
        
        # Plot 1: Ankle torques
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.time, self.torques['left_ankle'].values, 'b-', linewidth=2, alpha=0.3)
        ax1.plot(self.time, self.torques['right_ankle'].values, 'r-', linewidth=2, alpha=0.3)
        ax1.plot(self.time[:frame_idx+1], self.torques['left_ankle'].values[:frame_idx+1], 
                'b-', linewidth=2.5, label='Left')
        ax1.plot(self.time[:frame_idx+1], self.torques['right_ankle'].values[:frame_idx+1], 
                'r-', linewidth=2.5, label='Right')
        ax1.axvline(current_time, color='black', linestyle='--', linewidth=2)
        ax1.scatter([current_time], [self.torques['left_ankle'].values[frame_idx]], 
                   c='blue', s=100, zorder=5)
        ax1.scatter([current_time], [self.torques['right_ankle'].values[frame_idx]], 
                   c='red', s=100, zorder=5)
        ax1.set_ylabel('Ankle\nTorque (Nm)', fontsize=10, fontweight='bold')
        ax1.set_xlim(0, self.time[-1])
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)
        
        # Plot 2: Knee torques
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(self.time, self.torques['left_knee'].values, 'b-', linewidth=2, alpha=0.3)
        ax2.plot(self.time, self.torques['right_knee'].values, 'r-', linewidth=2, alpha=0.3)
        ax2.plot(self.time[:frame_idx+1], self.torques['left_knee'].values[:frame_idx+1], 
                'b-', linewidth=2.5)
        ax2.plot(self.time[:frame_idx+1], self.torques['right_knee'].values[:frame_idx+1], 
                'r-', linewidth=2.5)
        ax2.axvline(current_time, color='black', linestyle='--', linewidth=2)
        ax2.scatter([current_time], [self.torques['left_knee'].values[frame_idx]], 
                   c='blue', s=100, zorder=5)
        ax2.scatter([current_time], [self.torques['right_knee'].values[frame_idx]], 
                   c='red', s=100, zorder=5)
        ax2.set_ylabel('Knee\nTorque (Nm)', fontsize=10, fontweight='bold')
        ax2.set_xlim(0, self.time[-1])
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)
        
        # Plot 3: Hip pitch torques
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(self.time, self.torques['left_hip_pitch'].values, 'b-', linewidth=2, alpha=0.3)
        ax3.plot(self.time, self.torques['right_hip_pitch'].values, 'r-', linewidth=2, alpha=0.3)
        ax3.plot(self.time[:frame_idx+1], self.torques['left_hip_pitch'].values[:frame_idx+1], 
                'b-', linewidth=2.5)
        ax3.plot(self.time[:frame_idx+1], self.torques['right_hip_pitch'].values[:frame_idx+1], 
                'r-', linewidth=2.5)
        ax3.axvline(current_time, color='black', linestyle='--', linewidth=2)
        ax3.scatter([current_time], [self.torques['left_hip_pitch'].values[frame_idx]], 
                   c='blue', s=100, zorder=5)
        ax3.scatter([current_time], [self.torques['right_hip_pitch'].values[frame_idx]], 
                   c='red', s=100, zorder=5)
        ax3.set_ylabel('Hip Pitch\nTorque (Nm)', fontsize=10, fontweight='bold')
        ax3.set_xlim(0, self.time[-1])
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=8)
        
        # Plot 4: Total power
        ax4 = fig.add_subplot(gs[3])
        
        # Compute instantaneous power
        left_ankle_power = (self.torques['left_ankle'].values * 
                           self.qvel['left_ankle_vel'].values)
        right_ankle_power = (self.torques['right_ankle'].values * 
                            self.qvel['right_ankle_vel'].values)
        total_power = left_ankle_power + right_ankle_power
        
        ax4.plot(self.time, total_power, 'k-', linewidth=2, alpha=0.3)
        ax4.plot(self.time[:frame_idx+1], total_power[:frame_idx+1], 'k-', linewidth=2.5)
        ax4.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax4.axvline(current_time, color='black', linestyle='--', linewidth=2)
        ax4.scatter([current_time], [total_power[frame_idx]], c='black', s=100, zorder=5)
        ax4.fill_between(self.time[:frame_idx+1], 0, total_power[:frame_idx+1],
                        where=(total_power[:frame_idx+1] > 0), 
                        alpha=0.3, color='green', label='Generation')
        ax4.fill_between(self.time[:frame_idx+1], 0, total_power[:frame_idx+1],
                        where=(total_power[:frame_idx+1] < 0), 
                        alpha=0.3, color='red', label='Absorption')
        ax4.set_ylabel('Ankle\nPower (W)', fontsize=10, fontweight='bold')
        ax4.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        ax4.set_xlim(0, self.time[-1])
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=8)
        
        # Add time display
        fig.text(0.5, 0.98, f'Time: {current_time:.2f}s / {self.time[-1]:.2f}s', 
                ha='center', fontsize=12, fontweight='bold')
        
        # Convert to numpy array
        fig.canvas.draw()
        overlay_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        overlay_array = overlay_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return overlay_array
    
    def create_combined_frame(self, frame_idx):
        """Combine robot view and torque overlay side-by-side"""
        # Get robot view
        robot_view = self.capture_robot_view(frame_idx)
        
        # Get torque overlay
        torque_overlay = self.create_torque_overlay(frame_idx)
        
        # Combine horizontally
        combined = np.hstack([robot_view, torque_overlay])
        
        return combined
    
    def generate_frames(self):
        """Generate all frames with robot + torque data"""
        print(f"\n🎬 Generating {self.n_samples} frames with torque overlays...")
        
        for idx in range(self.n_samples):
            # Set robot state
            base_pos = [
                self.qpos['pelvis_x'].iloc[idx],
                self.qpos['pelvis_y'].iloc[idx],
                self.qpos['pelvis_z'].iloc[idx]
            ]
            base_orn = [
                self.qpos['pelvis_qx'].iloc[idx],
                self.qpos['pelvis_qy'].iloc[idx],
                self.qpos['pelvis_qz'].iloc[idx],
                self.qpos['pelvis_qw'].iloc[idx]
            ]
            
            p.resetBasePositionAndOrientation(self.robot_id, base_pos, base_orn)
            
            # Set joint states
            joint_mapping = {
                'left_hip_yaw': 'left_hip_yaw',
                'left_hip_roll': 'left_hip_roll',
                'left_hip_pitch': 'left_hip_pitch',
                'left_knee': 'left_knee',
                'left_ankle': 'left_ankle',
                'right_hip_yaw': 'right_hip_yaw',
                'right_hip_roll': 'right_hip_roll',
                'right_hip_pitch': 'right_hip_pitch',
                'right_knee': 'right_knee',
                'right_ankle': 'right_ankle',
                'torso': 'torso',
            }
            
            for qpos_col, joint_name in joint_mapping.items():
                if joint_name in self.joint_name_to_id:
                    joint_id = self.joint_name_to_id[joint_name]
                    position = self.qpos[qpos_col].iloc[idx]
                    velocity = self.qvel[f'{qpos_col}_vel'].iloc[idx]
                    p.resetJointState(self.robot_id, joint_id, position, velocity)
            
            p.stepSimulation()
            
            # Create combined frame
            combined_frame = self.create_combined_frame(idx)
            
            # Save frame
            frame_path = os.path.join(self.frame_dir, f"frame_{idx:04d}.png")
            Image.fromarray(combined_frame.astype(np.uint8)).save(frame_path)
            
            if idx % 5 == 0:
                print(f"  Progress: {idx+1}/{self.n_samples} ({100*(idx+1)/self.n_samples:.1f}%)", end='\r')
        
        print(f"\n✅ Generated {self.n_samples} frames in {self.frame_dir}/")
    
    def disconnect(self):
        p.disconnect()

# Usage
if __name__ == "__main__":
    H1_URDF_PATH = "/home/tilak/A/pybullet/unitree_ros/robots/h1_description/urdf/h1.urdf"
    
    try:
        print("="*60)
        print("H1 TORQUE VIDEO GENERATOR")
        print("="*60)
        
        generator = H1TorqueVideoGenerator(
            H1_URDF_PATH,
            frame_dir="h1_torque_frames"
        )
        
        generator.load_data(
            qpos_file='h1_qpos_labeled_2.csv',
            qvel_file='h1_qvel_labeled_2.csv',
            torques_file='h1_torques_labeled_2.csv',
            ankle_file='h1_ankle_processed_2.csv'
        )
        
        generator.generate_frames()
        generator.disconnect()
        
        print("\n" + "="*60)
        print("FRAMES GENERATED!")
        print("="*60)
        print("\nCreate video with:")
        print("  # Standard speed (50 fps):")
        print("  ffmpeg -framerate 50 -i h1_torque_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p h1_with_torques.mp4")
        print("\n  # Slow motion (25 fps):")
        print("  ffmpeg -framerate 25 -i h1_torque_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p h1_with_torques_slow.mp4")
        print("\n  # Very slow (10 fps):")
        print("  ffmpeg -framerate 10 -i h1_torque_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p h1_with_torques_veryslow.mp4")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
