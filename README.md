# Ankle Gait Stability In Humanoids

## MPX Setup Instructions

1) Install Conda (if not already installed)
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
2) Create the MPX Conda environment
```
conda create -n mpx_env python=3.12 -y
conda activate mpx_env
```
3) Install MuJoCo (Inside your Conda Env)
```
pip install mujoco==3.1.5
```
4) Install JAX (CPU mode if GPU is not available)
```
pip install --upgrade "jax[cpu]"
export JAX_PLATFORMS=cpu
```
Add the export to your bash 
```
echo "export JAX_PLATFORMS=cpu" >> ~/.bashrc
```
5) Follow the official installation steps from the original repo

    https://github.com/iit-DLSLab/mpx

6) Enable Logging in mjx_h1.py file (for now we are using the h1 robot for testing)
- Add these lists at the top of the simulation loop
```
  torque_log = []
  qpos_log = []
  qvel_log = []
```
- Inside the MPC update block, append the data into an array
```
tau, q, dq = mpc.run(qpos, qvel, input, contact)

torque_log.append(np.array(tau))
qpos_log.append(np.array(qpos))
qvel_log.append(np.array(qvel))
```
- Save the logs at the end
```
import numpy as np

np.save("h1_torque_log.npy", np.array(torque_log))
np.save("h1_qpos_log.npy", np.array(qpos_log))
np.save("h1_qvel_log.npy", np.array(qvel_log))

print("Saved logs: h1_torque_log.npy, h1_qpos_log.npy, h1_qvel_log.npy")
```
7) Run the mjx_h1.py file
```
python mpx/examples/mjx_h1.py
```
8) Analysis Scripts

This repo also contains scripts for converting .npy → .csv and visualizing the ankle behavior:

(Change the directories to save the logs accordingly) 

``` Analysis/test_logs.py ``` : Convert raw logs into labeled CSV

``` Analysis/test_analysis.py ``` : Generates -

- Ankle angle time series
- Torque profiles
- Phase portraits
- Gait stability metrics

