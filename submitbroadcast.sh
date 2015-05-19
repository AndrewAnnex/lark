#!/bin/bash
#PBS -S /bin/bash
#PBS -N Broadcast_Testing
#PBS -V
#PBS -e broadcast.log
#PBS -o broadcast.log
#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:01:00
cd $PBS_O_WORKDIR
/home/jlaura/anaconda/bin/mpirun -n 8 /home/jlaura/anaconda/bin/python broadcast.py
