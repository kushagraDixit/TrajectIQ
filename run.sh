#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
#SBATCH --mem=48GB
#SBATCH -o traject_iq-%j
#SBATCH --export=ALL

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/uufs/chpc.utah.edu/common/home/u1472614/miniconda3/lib
export MODEL='bert'
export SEED=0
export RES=9
export DIR_PATH='/scratch/general/vast/u1472614/trajectiq_runs/test_run2/'
export SCRDIR="/scratch/general/vast/$USER/trajectiq_runs/logs/$SLURM_JOBID"
export WORKDIR="$HOME/WORK/data_with_ml/TrajectIQ"



mkdir -p $SCRDIR
cp -r $WORKDIR/* $SCRDIR
cd $SCRDIR


source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/general/vast/u1472614/conda_envs/env_tiq
export MODE="preprocessing"
python ./traject_iq.py --model $MODEL --seed $SEED  --res $RES --dir_path $DIR_PATH --mode $MODE > my_out_1

export MODE="pretraining"
python ./traject_iq.py --model $MODEL --seed $SEED  --res $RES --dir_path $DIR_PATH --mode $MODE > my_out_2

export MODE="evaluation"
python ./traject_iq.py --model $MODEL --seed $SEED  --res $RES --dir_path $DIR_PATH --mode $MODE > my_out_3

