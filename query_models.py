import json
from typing import *
import argparse
from utils.globals import *
from utils.prompt_utils import AmbigousARCDataset
from utils.query import run_experiment, model_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the multiple choice experiment')

    parser.add_argument('--items_file', type=str, default=ITEMS_FILE, help='Path to the items file')
    parser.add_argument('--question_type', type=str, default='multiple_choice', help='Question type')
    parser.add_argument('--random_mat', type=bool, default=False, help='Randomize the matrix', action=argparse.BooleanOptionalAction)
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--system_prompt', type=str, default="You are a helpful assistant that solves analogy making puzzles. Only give the answer, no other words or text.", help='System prompt')
    parser.add_argument('--max_tokens', type=int, default=5, help='Max tokens')
    parser.add_argument('--logprobs', type=bool, default=True, help='Logprobs')
    parser.add_argument('--concept_answer_n', type=int, default=None, help='Number of concept answers')
    parser.add_argument('--results_file', type=str, help='Path to the results file')
    parser.add_argument('--example_item', type=bool, default=True, help='Example item')

    args = parser.parse_args()
    
    items_data = json.load(open(args.items_file, 'rb'))
    dataset = AmbigousARCDataset(
        items_data=items_data,
        batch_size=args.batch_size,
        question_type=args.question_type,
        random_mat=args.random_mat,
        example_item=args.example_item,
        concept_answer_n=args.concept_answer_n,
        seed=args.seed
    )
    print(dataset.x[0])

    # Run the experiment
    results = run_experiment(
        input_prompts=dataset.x, 
        models=model_list, 
        system_prompt=args.system_prompt,
        logprobs=args.logprobs,
        max_tokens=args.max_tokens,
        intermediate_results_path=args.results_file
    )