#!/bin/bash

models=(
    'Qwen/Qwen1.5-72B-Chat'
    'Qwen/Qwen1.5-110B-Chat'
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
    'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo'
    'google/gemma-2-9b-it'
    'google/gemma-2-27b-it'
)

python3 query_models.py \
    --models "${models[@]}" \
    --concept_answer_n 4 \
    --random_mat \
    --find_best_seed True \
    --duplicate "c" \
    --n_mirror 4 \
    --example_item \
    --encoding int \
    --results_dir data/results/run_2024-10-23