#!/usr/bin/env python3
"""Script for generating spinningup_experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing depending on the cluster:
SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

# NEED TO SET ALGO, ENV AND DATA_DIR HERE
ALGO = 'sac'
ENV='LunarLanderContinuous-v2'
DATA_DIR = f'{SCRATCH_HOME}/spinningup/data/{ALGO}'

if not os.path.exists(f'{DATA_DIR}'):
	os.makedirs(f'{DATA_DIR}')

base_call = f"python -m spinup.run {ALGO}_pytorch --env_name {ENV} --cpu 1 --epochs 50 --seed 0 10 20 --data_dir {DATA_DIR}"

alpha = [0.1, 0.2, 0.4]
batch_size = [100, 200]

settings = [(a, bs) for a in alpha for bs in batch_size]

output_file = open("spinningup_experiment.txt", "w")

for a, bs in settings:
	expt_call = (
		f"{base_call} "
		f"--exp_name {ALGO}_alpha{a}_batch_size{bs} "
		f"--alpha {a} "
		f"--batch_size {bs}"
	)
	print(expt_call, file=output_file)

output_file.close()
