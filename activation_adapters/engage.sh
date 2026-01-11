#!/usr/bin/env sh
#SBATCH -p mit_normal_gpu      # partition name
#SBATCH --job-name=activations_lowrank             # name for your job
#SBATCH --gres=gpu:1                 # if you need GPUs
#SBATCH --ntasks=1                   # number of tasks (often 1 for serial jobs)
#SBATCH --cpus-per-task=2            # CPU cores per task
#SBATCH --mem=64G                    # memory per node
#SBATCH --time=6:00:00              # max walltime (HH:MM:SS)
#SBATCH --output=sbatch_logs/act_lr-%j.out        # output file (%j = job ID) to capture logs for debugging




# python huggingface_llama.py
# python lora.py
python activations.py