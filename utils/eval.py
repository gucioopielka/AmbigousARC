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
    
def elementwise_match(list1: List[Any], list2: List[Any]) -> List[int]:
    return [1 if i == j else 0 for i, j in zip(list1, list2)]

def get_all_zero_indices(*lists: List[int]) -> List[int]:
    return [1 if all(x == 0 for x in items) else 0 for items in zip(*lists)]

# def accuracy(y_true: List[Any], y_pred: List[Any]) -> float:
#     assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be equal"
#     return sum(get_correct_responses(y_true, y_pred))/len(y_true)

class ModelEval():
    def __init__(
        self, 
        data: List[Dict],
        name: str,
        dataset: AmbigousARCDataset,
        prepended_symbols: List[str] = ['', '(', ' '],
        append_symbols: List[str] = []
    ):
        self.dataset = dataset
        self.data = data
        self.name = name

        if self.dataset.task == 'generation':  
            # Clean the responses
            self.clean_responses = [self.clean_generation_response(response['message']) for response in self.data]
            self.valid_responses_n = len([i for i in self.clean_responses if i])
            
            # The logprobs are the probabilities of each predicted digit in the response
            self.logprobs = []            
            for response in self.data:
                item_logprobs = []
                for token, logprob in zip(response['tokens'], response['logprobs']):
                    if token.isdigit():
                        item_logprobs.append(np.exp(logprob))
                self.logprobs.append(item_logprobs)
            
            # Classify the responses
            self.matrix_responses = elementwise_match([i[0] for i in self.dataset.y], self.clean_responses)
            self.concept_responses = elementwise_match([i[1] for i in self.dataset.y], self.clean_responses)
            self.duplicate_responses = self.get_dup_gen_responses(self.dataset.input_matrices, self.clean_responses)
            self.other_responses = get_all_zero_indices(self.matrix_responses, self.concept_responses, self.duplicate_responses)
            
            # Calculate the props
            self.matrix_prop = np.mean(self.matrix_responses)
            self.concept_prop = np.mean(self.concept_responses)
            self.duplicate_prop = np.mean(self.duplicate_responses)
            self.other_prop = np.mean(self.other_responses)
            
        if self.dataset.task in ['discrimination', 'recognition']:
            # Get the possible tokens
            self.possible_tokens = self.get_possible_tokens(prepended_symbols, append_symbols)
            
            # Clean the responses and get the logprobs
            self.clean_responses = []
            self.logprobs = []             
            self.choices = [] if self.dataset.task == 'discrimination' else None
            for response, y in zip(self.data, self.dataset.y):
                # Clean the response            
                response_clean, token_idx = self.clean_mc_response(response['tokens'])
                self.clean_responses.append(response_clean)                
                self.logprobs.append(np.exp(response['logprobs'][token_idx]) if response_clean else None) # logprobs for the answer choice
                
                if self.dataset.task == 'discrimination':
                    # Convert the response to the index of the answer choice (0: matrix, 1: concept, 2: random)
                    self.choices.append(y.index(response_clean) if response_clean else None)
            
            self.valid_responses_n = len([i for i in self.clean_responses if i])

            # Props and responses
            if self.dataset.task == 'discrimination':
                # Matrix: 0 | Concept: 1 | Random: 2 | Duplicate: 3
                self.matrix_responses = elementwise_match([0]*len(self.clean_responses), self.choices)
                self.concept_responses = elementwise_match([1]*len(self.clean_responses), self.choices)
                self.other_responses = elementwise_match([2]*len(self.clean_responses), self.choices)
                self.duplicate_responses = elementwise_match([3]*len(self.clean_responses), self.choices)
                
                # Calculate the props
                self.matrix_prop = np.mean(self.matrix_responses)
                self.concept_prop = np.mean(self.concept_responses)
                self.other_prop = np.mean(self.other_responses)
                self.duplicate_prop = np.mean(self.duplicate_responses)

            elif self.dataset.task == 'recognition':
                self.concept_responses = elementwise_match(self.dataset.y, self.clean_responses)
                self.other_responses = get_all_zero_indices(self.concept_responses)
                self.concept_prop = np.mean(self.concept_responses)
                self.other_prop = np.mean(self.other_responses)

            # The same answer choice as example
            example_y = [i[1] for i in self.dataset.example_y] if self.dataset.task == 'discrimination' else self.dataset.example_y
            self.same_as_example = elementwise_match(example_y, self.clean_responses)

            # Answer choices proportions
            self.answer_props = {}
            for answer_opt in self.dataset.answer_options:
                if self.valid_responses_n > 0:
                    self.answer_props[answer_opt] = np.mean(elementwise_match([answer_opt]*len(self.clean_responses), self.clean_responses))
                else:
                    self.answer_props[answer_opt] = None

            # Proportion test 
            if self.valid_responses_n > 0:
                observed = [sum([i == answer for i in self.clean_responses]) for answer in self.dataset.answer_options]
                expected = [len([i for i in self.clean_responses if i]) / len(self.dataset.answer_options)] * len(self.dataset.answer_options)
                self.prop_test_p = self.prop_test(observed, expected)
            else:
                self.prop_test_p = None

    def get_dup_gen_responses(self, input_matrices, responses):
        return [1 if response in input else 0 for input, response in zip(input_matrices, responses)]
        
    def get_possible_tokens(self, prepended_symbols: List[str], append_symbols: List[str]) -> List[str]:
        possible_tokens = []
        for i in self.dataset.answer_options:
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
            return [i for i in self.dataset.answer_options if i in choice][0], tokens.index(choice)
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
        data: str,
        dataset: AmbigousARCDataset = None,
        prop_test_thresh: float = 0.05,
        no_response_thresh: float = 0.75
    ):
        # Load the results
        data = json.load(open(data, 'rb'))
        self.results = data['data'] 
        dataset_config = data['dataset_config']

        self.dataset = AmbigousARCDataset(**dataset_config) if dataset_config else dataset

        self.question_type = self.dataset.task  
        self.all_models_n = len(self.results)
        self.models_names = list(self.results.keys())
        
        self.models = [ModelEval(self.results[model], model, self.dataset) for model in self.models_names]
        
        # Exclude models with high no response rate 
        self.excluded_models_no_response = []
        if no_response_thresh:
            self.excluded_models_no_response = [model for model in self.models if len([i for i in model.clean_responses if i]) < int(no_response_thresh*self.dataset.size)]
            self.models = [model for model in self.models if len([i for i in model.clean_responses if i]) >= int(no_response_thresh*self.dataset.size)]
            self.models_names = [model.name for model in self.models]
        
        # Exclude models with high prop test
        if self.question_type != 'generation':
            self.excluded_models_prop_test = []
            if prop_test_thresh:
                self.excluded_models_prop_test = [model for model in self.models if model.prop_test_p and model.prop_test_p < prop_test_thresh]
                self.models = [model for model in self.models if model.prop_test_p and model.prop_test_p >= prop_test_thresh]
                self.models_names = [model.name for model in self.models]

        # Get the response metrics
        # self.concept_props = [model.concept_responses for model in self.models]
        # self.matrix_props = [model.matrix_responses for model in self.models]
        # self.other_responses = [model.other_responses for model in self.models]

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
            'item_main_id': np.tile([i.split('_')[0] for i in self.dataset.item_ids], len(self.models)),
            'item_id': np.tile(self.dataset.item_ids, len(self.models)),
            'model': np.repeat(self.models_names, self.dataset.size),
            'response': np.concatenate([model.clean_responses for model in self.models]),
            'concept': np.tile(self.dataset.concepts, len(self.models)),
        }
        if self.dataset.task == 'discrimination':
            df_config['choice'] = [{0: 'matrix', 1: 'concept', 2: 'other', 3: 'duplicate'}.get(i) for i in np.concatenate([model.choices for model in self.models])] 
        df_config['concept_response'] = np.concatenate([model.concept_responses for model in self.models]) 
        df_config['other_response'] = np.concatenate([model.other_responses for model in self.models])
        if self.dataset.task != 'recognition':
            df_config['matrix_response'] = np.concatenate([model.matrix_responses for model in self.models])
            df_config['duplicate_response'] = np.concatenate([model.duplicate_responses for model in self.models])
        
        if self.dataset.task == 'generation':
            df_config['logprobs'] = np.concatenate([[np.mean(logprobs) for logprobs in model.logprobs] for model in self.models])
        else:
            df_config['logprobs'] = np.concatenate([model.logprobs for model in self.models])
            df_config['same_as_example'] = np.concatenate([model.same_as_example for model in self.models])

        return pd.DataFrame(df_config)