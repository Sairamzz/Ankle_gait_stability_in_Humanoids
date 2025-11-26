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
