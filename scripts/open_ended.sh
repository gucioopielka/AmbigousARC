#!/bin/bash
cd ..

python3 query_models.py \
    --question_type open_ended \
    --seed 42 \
    --results_file data/results/open_ended.json \
    --max_tokens 100 \
