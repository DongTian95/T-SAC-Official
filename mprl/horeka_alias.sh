# Author: Ge Li, ge.li@kit.edu
# Description: Alias for Horeka HPC experiments, I used these alias to run experiments quickly on Horeka HPC.
# Usage: source horeka_alias.sh
# Note: You can add and adapt your own alias / path here, and source this file in your .bashrc file to use these alias commands.

# Horeka: https://www.scc.kit.edu/en/services/horeka.php

# Slurm watch alias
alias wa='watch -n 20 squeue'
alias was='watch -n 20 squeue --start'
alias keep='watch -n 600 squeue'
alias sfree='sinfo_t_idle'
alias sgpu='salloc -p accelerated -n 1 -t 120 --mem=20000 --gres=gpu:1'
alias scpu='salloc -p cpuonly -n 1 -t 120 --mem=20000'

# Git alias
alias gp='git pull'

# clear workspace
alias clc="rm -rf ./log && rm -rf ../mprl_exp_result && rm -rf MPRL.egg-info && rm -rf ../../CODE_COPY"
alias idle="sinfo_t_idle"
alias usage="sreport -t Hours cluster UserUtilizationByAccount Start=2025-05-01 End=now user=lb7471 --tres cpu,gres/gpu"

# Env alias
alias vb='cd ~/ && vim .bashrc'
alias ss='cd ~/ && source .bashrc && conda activate tsacOfficial && module load devel/cuda/11.8'
alias sstp="cd ~/ && source .bashrc && conda activate fabian_sac && module load devel/cuda/11.8 && export CC=gcc && export CXX=g++"
alias cdws='cd ~/dt_rl/mprl'
alias cque="squeue -u lb7471"

# Exp
alias runexp='python mp_exp.py'
alias runexp_mp='python mp_exp_multiprocessing.py'

## DRP BOX PUSHING RANDOM INIT
alias bp_drp='runexp ./config/box_push_random_init/prodmp_replan_drp_nanogpt/horeka.yaml -o -s'
alias bp_tsac='runexp ./config/box_push_random_init/transformer_sac/horeka.yaml -o -s'
alias bpmp_tsac='runexp_mp ./config/box_push_random_init/transformer_sac_multiprocessing/horeka.yaml -o -s'
alias bpsmp_tsac='runexp_mp ./config/box_push_random_init_sparse/transformer_sac_multiprocessing/horeka.yaml -o -s'
alias bpmp_rlac_tc='runexp_mp ./config/box_push_random_init/rlac_tc/horeka.yaml -o -s'

## gym Sets
alias gymmp_rlac_tc='runexp_mp ./config/gym_mujoco/rlac_tc/horeka.yaml -o -s'

## MetaWorld
alias metaworld_tsac='runexp ./config/metaworld/transformer_sac_cpu/horeka.yaml -o -s'
alias metaworldmp_tsac='runexp_mp ./config/metaworld/transformer_sac_multiprocessing/horeka.yaml -o -s'

## nrsac
alias bp_nrsac='runexp ./config/box_push_random_init/n_returns_sac/cpuonly.yaml -o -s'
alias metaworld_nrsac='runexp ./config/metaworld/n_returns_sac/cpuonly.yaml -o -s'

## Gymnasium Pendulum for testing
alias pendulum_cpu='runexp ./config/pendulum/n_returns_sac/cpuonly.yaml -o -s'

## Weiran's experiments
alias cdws_wr="cd /hkfs/work/workspace/scratch/lb7471-blah/wr_exp"