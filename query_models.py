import os
import json
from datetime import date
import warnings
from typing import *
import argparse
from utils.globals import ITEMS_FILE, RESULTS_DIR
from utils.prompt_utils import AmbigousARCDataset
from utils.query import run_experiment
from together import Together
from openai import OpenAI

together_client = Together()
openai_client = OpenAI()

together_model_list = [model.id for model in together_client.models.list() if model.type == 'chat']
openai_model_list = []#TODO
full_model_list = together_model_list + openai_model_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the multiple choice experiment')

    parser.add_argument('--items_file', type=str, default=ITEMS_FILE, help='Path to the items file')
    parser.add_argument('--models', nargs='+', default=[], type=str, help='Models to run the experiment on')
    parser.add_argument('--random_mat', type=bool, help='Randomize the matrix', action=argparse.BooleanOptionalAction)
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help='Random seed')
    parser.add_argument('--find_best_seed', type=bool, default=False, help='Find the best seed')
    parser.add_argument('--n_iter_seed', type=int, default=100, help='Number of iterations to find the best seed')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--n_mirror', type=int, help='Number of mirror items')
    parser.add_argument('--concept_answer_n', type=int, help='Number of concept answers')
    parser.add_argument('--duplicate', type=str, default=None, help='Duplicate type')
    parser.add_argument('--encoding', type=str, default='int', help='Encoding type (int or color)')
    parser.add_argument('--results_dir', type=str, help='Results directory')
    parser.add_argument('--example_item', type=bool, default=True, help='Example item', action=argparse.BooleanOptionalAction)
    parser.add_argument('--tasks', nargs='+', default=['generation', 'discrimination', 'recognition'], type=str, help='Tasks to run the experiment on')

    args = parser.parse_args()
        

    if args.models:
        for model in args.models:
            if model not in full_model_list:
                raise ValueError(f'Model "{model}" is not available')

    if args.seeds and len(args.seeds) != len(args.tasks):
        raise ValueError('The number of seeds should be equal to the number of tasks')

    if args.seeds and args.find_best_seed:
        warnings.warn('Both seeds and find_best_seed are provided. The seeds will be ignored.')    
    
    elif not args.seeds and not args.find_best_seed:
        warnings.warn('No seed is provided. The results will be non-deterministic.')

    if args.concept_answer_n != (0 if args.duplicate is None else 1 + args.random_mat + 2):
        warnings.warn('The number of multiple choice answer options in recognition and discrimination tasks is not equal.')
    

    for idx, task in enumerate(args.tasks):
        print(f'\n{'-_'*5}Running task: {task}{'_-'*5}\n')

        # Create the results directory
        results_dir = args.results_dir if args.results_dir else os.path.join(RESULTS_DIR, f'run_{date.today()}')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_file = os.path.join(results_dir, f'{task}.json')

        # Create the dataset
        dataset = AmbigousARCDataset(
            task = task,
            items_data=args.items_file,
            batch_size=args.batch_size,
            random_mat = args.random_mat,
            example_item = args.example_item,
            n_mirror = args.n_mirror,
            concept_answer_n = args.concept_answer_n,
            duplicate = args.duplicate,
            encoding = args.encoding,
            seed = args.seeds[idx] 
                if args.seeds 
                else None,
            find_best_seed = args.find_best_seed 
                if task != 'generation'
                else False,
            n_iter_seed = args.n_iter_seed
        )
        
        # Run the experiment
        results = run_experiment(
            client = together_client,
            input_prompts=dataset.x, 
            logprobs = True,
            models = args.models 
                if args.models 
                else full_model_list, 
            system_prompt = "What is the concept that best describes the following task? Only give the answer, no other words or text."
                if task == 'recognition' 
                else "You are a helpful assistant that solves analogy making puzzles. Only give the answer, no other words or text.",
            max_tokens = 100 
                if task == 'generation' 
                else 5,
            results_path = results_file
        )

        # Save the dataset configuration
        results['dataset_config'] = dataset.get_config()
        json.dump(results, open(results_file, 'w'), indent=4)