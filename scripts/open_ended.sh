#!/bin/bash
cd ..

python3 query_models.py \
    --question_type generation \
    --seed 42 \
    --results_file data/results/open_ended.json \
    --max_tokens 100 \
