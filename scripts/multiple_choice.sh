#!/bin/bash
cd ..
# python3 query_models.py \
#     --question_type multiple_choice \
#     --no-random_mat \
#     --seed 42 \
#     --results_file data/results/multiple_choice.json \

# python3 query_models.py \
#     --question_type multiple_choice \
#     --random_mat \
#     --seed 42 \
#     --results_file data/results/multiple_choice_random.json \

# Seeds chosen to make the answers options as random as possible
python3 query_models.py \
    --question_type multiple_choice \
    --no-random_mat \
    --seed 5 \
    --results_file data/results/multiple_choice.json \

python3 query_models.py \
    --question_type multiple_choice \
    --random_mat \
    --seed 45 \
    --results_file data/results/multiple_choice_random.json \
