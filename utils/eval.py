from typing import *
from utils.prompt_utils import AmbigousARCDataset
import numpy as np
import pandas as pd
from rich import print as rprint
from scipy.stats import chisquare
import re
import json

def delimit_response(response: str) -> str:
    return response.replace('[', '').replace(']', '').replace(' ', '')

def filter_response(s: str) -> str:
    """
    Extracts and returns a substring containing the first valid sequence found in the input string.
    The valid sequence is defined as either three consecutive '[int int int]' patterns or
    three consecutive '[int int int int int]' patterns. Newline characters in the
    extracted substring are removed.

    If no valid sequence is found, returns numpy.nan.
    """
    pattern = r'((\[\s*\d+\s+\d+\s+\d+\s*\]\s*){3}|(\[\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s*\]\s*){5})'
    match = re.search(pattern, s)
    if match:
        result = s[match.start():match.end()]
        return result.replace('\n', '')
    else:
        return None
    
def get_correct_responses(y_true: List[Any], y_pred: List[Any]) -> List[int]:
    return [i == j for i, j in zip(y_pred, y_true)]

def accuracy(y_true: List[Any], y_pred: List[Any]) -> float:
    assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be equal"
    return sum(get_correct_responses(y_true, y_pred))/len(y_true)

class ModelEval():
    def __init__(
        self, 
        results: List[Dict],
        name: str,
        dataset: AmbigousARCDataset,
        prepended_symbols: List[str] = ['', '(', ' '],
        append_symbols: List[str] = []
    ):
        self.dataset = dataset
        self.results = results
        self.name = name

        if self.dataset.task == 'generation':
            self.clean_responses = []
            self.logprobs = []
            # for response in results:
            #     clean_response = self.clean_generation_response(response['message'])
                
            #     self.clean_responses.append(clean_response)

            self.clean_responses = [self.clean_generation_response(response['message']) for response in results]
            self.logprobs = [response['logprobs'] for idx, response in enumerate(results)]
            self.matrix_responses = accuracy(np.array(self.dataset.y)[:, 0], self.clean_responses)
            self.concept_responses = accuracy(np.array(self.dataset.y)[:, 1], self.clean_responses)
            
        if self.dataset.task == 'discrimination' or self.dataset.task == 'recognition':
            # Prepare the possible tokens
            if self.dataset.task == 'discrimination':
                self.answer_choices = ['a', 'b', 'c'] if self.dataset.random_mat else ['a', 'b']
            else:
                self.answer_choices = [chr(97 + i) for i in range(len(set(self.dataset.y)))]
            self.possible_tokens = self.get_possible_tokens(prepended_symbols, append_symbols)
            
            # Clean the responses
            self.clean_responses = []
            self.logprobs = []             
            self.choices = [] if self.dataset.task == 'discrimination' else None

            for response, y in zip(results, self.dataset.y):            
                response_clean, token_idx = self.clean_mc_response(response['tokens'])
                self.clean_responses.append(response_clean)
                self.logprobs.append(response['logprobs'][token_idx] if response_clean else None) # logprobs for the answer choice
                
                if self.choices is not None:
                    # Convert the response to the index of the answer choice (0: matrix, 1: concept, 2: random)
                    self.choices.append(y.index(response_clean) if response_clean else None) # e.g. ['c', 'a', 'b'].index('a')

            # Answer choices proportions
            self.answer_props = {}
            for answer in self.answer_choices:
                if len([i for i in self.clean_responses if i]) > 0:
                    self.answer_props[answer] = sum([i == answer for i in self.clean_responses])/len([i for i in self.clean_responses if i])
                else:
                    self.answer_props[answer] = None

            if len([i for i in self.clean_responses if i]) > 0:
                observed = [sum([i == answer for i in self.clean_responses]) for answer in self.answer_choices]
                expected = [len([i for i in self.clean_responses if i]) / len(self.answer_choices)] * len(self.answer_choices)
                self.prop_test_p = self.prop_test(observed, expected)
            else:
                self.prop_test_p = None

            # Calculate the accuracy
            if self.dataset.task == 'discrimination':
                # Matrix: 0 | Concept: 1 | Random: 2
                self.matrix_responses = accuracy([0]*len(self.clean_responses), self.choices)
                self.concept_responses = accuracy([1]*len(self.clean_responses), self.choices)
                self.random_responses = accuracy([2]*len(self.clean_responses), self.choices) if self.dataset.random_mat else None            
            elif self.dataset.task == 'recognition':
                self.accuracy = accuracy(self.dataset.y, self.clean_responses)
    
            
    def get_possible_tokens(self, prepended_symbols: List[str], append_symbols: List[str]) -> List[str]:
        possible_tokens = []
        for i in self.answer_choices:
            for j in prepended_symbols:
                possible_tokens.append(j + i)
            for j in append_symbols:
                possible_tokens.append(i + j)            
        return possible_tokens
    
    def clean_mc_response(self, tokens: List[str]) -> Tuple[str, int]:       
        # check if any of the tokens is in the possible tokens
        choice = None
        for token in tokens:
            if token in self.possible_tokens:
                choice = token
                break
        
        # return the answer choice option
        if choice:
            return [i for i in self.answer_choices if i in choice][0], tokens.index(choice)
        else:
            return None, None
        
    def clean_generation_response(self, response: str, ) -> str:
        #TODO: Checking dimensionality of the response??
        response = filter_response(response)
        return delimit_response(response) if response else None
    
    def prop_test(self, observed, expected) -> float:
        return chisquare(f_obs=observed, f_exp=expected)[1]

class Eval():
    def __init__(
        self,
        results: Dict|str,
        dataset: AmbigousARCDataset = None,
        prop_test_thresh: float = 0.05,
        no_response_thresh: float = 0.75
    ):
        # Load the results
        if isinstance(results, dict):
            self.results = results
        else:
            self.results = json.load(open(results, 'r'))

        # Load the dataset
        if 'data_config' in self.results:
            dataset_config = self.results.pop('data_config')
        assert dataset or dataset_config, "Either dataset or dataset_config must be provided"

        if not dataset:
            self.dataset = AmbigousARCDataset(**dataset_config)
        else:
            self.dataset = dataset
        
        self.question_type = self.dataset.task  
        self.all_models_n = len(self.results)
        self.models_names = list(self.results.keys())
        self.models = [ModelEval(self.results[model], model, self.dataset) for model in self.models_names]
        self.excluded_models_no_response = []
        self.excluded_models_prop_test = []

        if no_response_thresh:
            # Exclude models with high no response rate
            self.excluded_models_no_response = [model for model in self.models if len([i for i in model.clean_responses if i]) < int(no_response_thresh*self.dataset.size)]
            self.models = [model for model in self.models if len([i for i in model.clean_responses if i]) >= int(no_response_thresh*self.dataset.size)]
            self.models_names = [model.name for model in self.models]
        
        if prop_test_thresh:
            # Exclude models with high prop test
            self.excluded_models_prop_test = [model for model in self.models if model.prop_test_p and model.prop_test_p < prop_test_thresh]
            self.models = [model for model in self.models if model.prop_test_p and model.prop_test_p >= prop_test_thresh]
            self.models_names = [model.name for model in self.models]

        if self.question_type == 'discrimination' or self.question_type == 'generation':     
            self.concept_responses = [model.concept_responses for model in self.models]
            self.matrix_responses = [model.matrix_responses for model in self.models]
            self.random_responses = [model.random_responses for model in self.models] if self.dataset.random_mat else None

        elif self.question_type == 'recognition':
            self.accuracy = [model.accuracy for model in self.models]

        self.df = self.to_pd()

    def __str__(self):
        s = f'Number of models: {len(self.models)}\n'
        if self.question_type == 'discrimination' or self.question_type == 'generation':
            if self.concept_responses and self.matrix_responses:
                s += f'Concept: {np.mean(self.concept_responses):.2f}\nMatrix: {np.mean(self.matrix_responses):.2f}'
            else: 
                s += 'No valid responses'
        elif self.question_type == 'recognition':
            s += f'Accuracy: {np.mean(self.accuracy):.2f}'
        if self.dataset.random_mat:
            s += f'\nRandom: {np.mean(self.random_responses):.2f}'
        if any([self.excluded_models_no_response, self.excluded_models_prop_test]):            
            s += f'\n\nTotal excluded models: {len(self.excluded_models_no_response) + len(self.excluded_models_prop_test)}/{self.all_models_n} ({((len(self.excluded_models_no_response) + len(self.excluded_models_prop_test))/self.all_models_n)*100:.0f}%)'
        if self.excluded_models_no_response:
            s += f'\n{len(self.excluded_models_no_response)}/{self.all_models_n} ({(len(self.excluded_models_no_response)/self.all_models_n)*100:.0f}%) models excluded due to high no response rate.'
        if self.excluded_models_prop_test:
            s += f'\n{len(self.excluded_models_prop_test)}/{self.all_models_n} ({(len(self.excluded_models_prop_test)/self.all_models_n)*100:.0f}%) models excluded due to biased answer choices.'
        return s
    
    def print(self):
        rprint(str(self))

    def to_pd(self):
        if len(self.models_names) == 0:
            print('No models to evaluate')
            return None
        
        df_config = {
            'item_id': np.tile(self.dataset.items, len(self.models)),
            'model': np.repeat(self.models_names, self.dataset.size),
            'response': np.concatenate([model.clean_responses for model in self.models]),
            'concept': np.tile(self.dataset.concepts, len(self.models)),
        }
        if self.question_type == 'discrimination':
            all_model_choices = np.concatenate([model.choices for model in self.models])
            df_config['choice'] = [{0: 'matrix', 1: 'concept', 2: 'random'}.get(i) for i in all_model_choices] 
            df_config['matrix_response'] = [1 if i == 0 else 0 for i in all_model_choices] 
            df_config['concept_response'] = [1 if i == 1 else 0 for i in all_model_choices] 
            df_config['random_response'] = [1 if i == 2 else 0 for i in all_model_choices]

        if self.question_type == 'recognition':
            # Get accuracy per item per model
            df_config['concept_acc'] = np.concatenate([get_correct_responses(self.dataset.y, model.clean_responses) for model in self.models]) 

        if self.dataset.task == 'generation':
            df_config['matrix_response'] = np.concatenate([get_correct_responses(np.array(self.dataset.y)[:, 0], model.clean_responses) for model in self.models])
            df_config['concept_response'] = np.concatenate([get_correct_responses(np.array(self.dataset.y)[:, 1], model.clean_responses) for model in self.models])
        
        if self.dataset.task == 'generation':
            df_config['logprobs'] = np.concatenate([[np.mean(logprobs) for logprobs in model.logprobs] for model in self.models])
        else:
            df_config['logprobs'] = np.concatenate([model.logprobs for model in self.models])

        return pd.DataFrame(df_config)