#!/bin/bash

# Load models from YAML file
yaml() { python3 -c "import yaml; print('\n'.join(yaml.safe_load(open('$1'))$2))" ; }
models=($(yaml globals.yml "['MODEL_IDS']"))

# Color mirror
printf "\nRunning color mirror...\n"
python3 query_models.py \
    --models "${models[@]}" \
    --concept_answer_n 4 \
    --random_mat \
    --find_best_seed \
    --duplicate c \
    --n_mirror 4 \
    --example_item different \
    --encoding int \
    --mirror_type color \
    --results_dir data/results/run_2024-11-12

# Example mirror
printf "\nRunning example mirror...\n"
python3 query_models.py \
    --models "${models[@]}" \
    --concept_answer_n 4 \
    --random_mat \
    --find_best_seed \
    --duplicate c \
    --n_mirror 4 \
    --example_item different \
    --encoding int \
    --mirror_type example \
    --results_dir data/results/run_example_2024-11-12

# # Color encoding
# printf "\nRunning color encoding...\n"
# python3 query_models.py \
#     --models "${models[@]}" \
#     --concept_answer_n 4 \
#     --random_mat \
#     --find_best_seed \
#     --duplicate c \
#     --n_mirror 4 \
#     --example_item different \
#     --encoding color \
#     --mirror_type color \
#     --results_dir data/results/run_ColorEncoding_2024-11-13