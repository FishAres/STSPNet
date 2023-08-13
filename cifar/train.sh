#!/bin/bash

seed_arr=($(seq 1 1 10))

# train
for seed in "${seed_arr[@]}"
do
    python main.py --seed=1 --save-model
done
