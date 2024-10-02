#!/bin/bash
cd ..

python3 query_models.py \
    --question_type concept_task \
    --seed 12 \
    --results_file data/results/concept_task_abcd.json \
    --system_prompt "What is the concept of the following task? Only give the answer, no other words or text." \
    --concept_answer_n 4

python3 query_models.py \
    --question_type concept_task \
    --seed 9123 \
    --results_file data/results/concept_task_all.json \
    --system_prompt "What is the concept of the following task? Only give the answer, no other words or text." \
    #--concept_answer_n "None" 