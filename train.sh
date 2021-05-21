#!/bin/bash

echo "Please choose your model: "
model_arr = ()
for entry in `ls /osim-rl/osim/models/`; do
    echo $entry
    model_arr += ($entry)
done

read model_idx

start_iter=0

while true; do
  mpirun --use-hwthread-cpus -np 4 python main.py $start_iter $model_arr[model_idx]
  wait
  start_iter=1
done

