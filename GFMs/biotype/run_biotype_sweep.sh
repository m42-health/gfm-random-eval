#!/bin/bash

MIN_GPU=0
MAX_GPU=8
SWEEP_ID="<PLACE YOUR SWEEP ID HERE>"

for ((i=MIN_GPU;i<MAX_GPU;i++)); do
    CUDA_VISIBLE_DEVICES=$i wandb agent $SWEEP_ID &
done
wait
