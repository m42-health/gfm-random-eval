#!/bin/bash

MIN_GPU=0
MAX_GPU=8
SWEEP_ID="<PLACE_YOUR_SWEEP_ID_HERE>"

for ((i=MIN_GPU;i<MAX_GPU;i++)); do
    CUDA_VISIBLE_DEVICES=$i wandb agent $SWEEP_ID &
done
wait
