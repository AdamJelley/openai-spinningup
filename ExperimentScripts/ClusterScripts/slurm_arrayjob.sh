#!/bin/bash
#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
# #SBATCH -N 1	  # nodes requested
# #SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:0  # use 0 GPU
#SBATCH --mem=10000  # memory in Mb
#SBATCH --partition=CDT_Compute
#SBATCH -t 2-00:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=1  # number of cpus to use - there are 20 on each node.

set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# ====================
# Activate Anaconda environment
# ====================
source /home/${USER}/miniconda3/bin/activate spinningup

# ====================
# RSYNC data from /home/ to /disk/scratch/
# ====================
export SCRATCH_HOME=/disk/scratch/${USER}
# export DATA_HOME=${PWD}/data
# export DATA_SCRATCH=${SCRATCH_HOME}/pgr/data
# mkdir -p ${SCRATCH_HOME}/pgr/data
# rsync --archive --update --compress --progress ${DATA_HOME}/ ${DATA_SCRATCH}

# ====================
# Run training. Here we use src/gpu.py
# ====================
echo "Creating directory to save outputs"
export OUTPUT_DIR=${SCRATCH_HOME}/spinningup/data/
mkdir -p ${OUTPUT_DIR}

# ===================
# Run the experiment
# ===================
# Read line number ${SLURM_ARRAY_TASK_ID} from the experiment file and run it
# ${SLURM_ARRAY_TASK_ID} is simply the number of the job within the array. If
# you execute `sbatch --array=1:100 ...` the jobs will get numbers 1 to 100
# inclusive.

experiment_text_file=$1
COMMAND="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" ${experiment_text_file}`"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"


# ====================
# Run prediction. We will save outputs and weights to the same location but this is not necessary
# ====================
# python src/gpu_predict.py \
# 	--data_path=${DATA_SCRATCH}/pred.txt \
# 	--model_dir=${OUTPUT_DIR} 

# ====================
# RSYNC data from /disk/scratch/ to /home/. This moves everything we want back onto the distributed file system
# ====================
OUTPUT_HOME=/home/s2139934/projects/spinningup/data/
mkdir -p ${OUTPUT_HOME}
rsync --archive --update --compress --progress ${OUTPUT_DIR} ${OUTPUT_HOME}

# ====================
# Finally we cleanup after ourselves by deleting what we created on /disk/scratch/
# ====================
rm -rf ${OUTPUT_DIR}

echo ""
echo "============"
echo "Job ${SLURM_JOB_ID} finished successfully!"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"

