#!/bin/bash

# Initialize Conda for this script
eval "$(conda shell.bash hook)"

# Define environment name and Python version
ENV_NAME=tsac_official_env
PYTHON_VERSION=3.11

# Create a new conda environment
echo "Creating a new conda environment named $ENV_NAME with Python $PYTHON_VERSION"
conda create --name $ENV_NAME python=$PYTHON_VERSION -y

# Activate the newly created environment
echo "Activating the $ENV_NAME environment"
conda activate $ENV_NAME

# Verify if the correct conda environment is activated
echo
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "$CONDA_DEFAULT_ENV"
    echo Failed to activate conda environment.
    exit 1
else
    echo Successfully activated conda environment.
fi

# Install mamba release to boost installation and resolve dependencies
# conda install -c conda-forge mamba=1.4.2 -y


# Install packages using conda or mamba
echo "Installing packages with conda or mamba"
conda install -c conda-forge conda-build -y

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install tqdm==4.67.3 wandb natsort==8.4.0 tabulate==0.9.0 "numpy<2" pytest psutil==7.2.2 addict==2.4.0 matplotlib==3.10.8

# Add the current TCE repo to python path
conda develop .

cd ..
# Download packages from Github

# Fancy_Gym
git clone -b dt_branch --single-branch git@github.com:DongTian95/fancy_gymnasium.git
cd fancy_gymnasium
pip uninstall fancy_gymnasium -y
conda develop .
cd ..

# Git_Repo_Tracker
git clone -b main --single-branch git@github.com:ALRhub/Git_Repos_Tracker.git
cd Git_Repos_Tracker
pip uninstall git_repos_tracker -y
conda develop .
cd ..

# MetaWorld
git clone -b bruce_dev --single-branch git@github.com:BruceGeLi/Metaworld.git
cd Metaworld
pip uninstall metaworld -y
conda develop .
cd ..

# cw2
git clone -b bruce_dev --single-branch git@github.com:BruceGeLi/cw2.git
cd cw2
pip uninstall cw2 -y
conda develop .
cd ..

# MP_PyTorch
git clone -b bruce_dev --single-branch git@github.com:ALRhub/MP_PyTorch.git
cd MP_PyTorch
pip uninstall mp_pytorch -y
conda develop .
cd ..

# Install packages using pip
echo "Installing packages with pip"
pip install stable-baselines3==2.2.1

echo "Configuration completed successfully."

