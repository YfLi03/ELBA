#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J timing
#SBATCH --mail-user=yl3722@cornell.edu
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 16 -c 16 --cpu_bind=cores /global/homes/y/yfli03/ELBA/elba -A 1 -B 1 -x 30 -c 0.65 /pscratch/sd/y/yfli03/ELBA_dataset/ecoli_hifi/reads.fa | tee result_timer_radix_sort.log