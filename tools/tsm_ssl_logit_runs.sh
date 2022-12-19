#!/bin/bash

for i in {2..6}
do
sbatch tools/tsm_slurm_$i.sh
done