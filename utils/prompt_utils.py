import numpy as np
import json
from typing import *
from utils.plot_utils import *

def convert_to_array(x: str, dim: int) -> np.array:
    return np.array([int(char) for char in x]).reshape(dim, dim)

def encode_array(array: np.array) -> str:
    return ''.join([str(x) for x in array.flatten()])

def item_to_arrays(
    item: str, 
    items_data: dict, 
    matrices: List[str] = ['A', 'B', 'C']
) -> List[np.array]:
    '''Returns a list of arrays for the given item'''
    assert item in items_data, f"Item '{item}' not found in items data"
    
    # Check for each matrix in the item's data
    missing_matrices = [mat for mat in matrices if mat not in items_data[item]]
    if missing_matrices:
        raise ValueError(f"Matrix (or matrices) {missing_matrices} not found in item '{item}'")
    
    return [convert_to_array(items_data[item][mat], items_data[item]['xdim']) for mat in matrices]

def int_to_color(int_encoding) -> str:
    int_to_color = {
        '0': "black",
        '1': "red",
        '2': "orange",
        '3': "yellow",
        '4': "green",
        '5': "blue",
        '6': "purple",
        '7': "pink",
    }
    return int_to_color[int_encoding]

def concept_to_verb(concept: str) -> str:
    concept_to_verb = {
        'above_and_below': 'above and below',
        'clean_up': 'clean up',
        'complete_shape': 'complete shape',
        'copy': 'copy',
        'extend_to_boundary': 'extend to boundary',
        'filled_and_not_filled': 'filled and not filled',
        'horizontal_and_vertical': 'horizontal and vertical',
        'inside_and_outside': 'inside and outside',
        'move_to_boundary': 'move to boundary',
        'order': 'order',
        'other': 'other',
        'top_and_bottom_2d': 'top and bottom 2D',
        'top_and_bottom_3d': 'top and bottom 3D',
    }
    return concept_to_verb[concept]


def array_to_str(array: np.array, encoding='int', show_brackets=True)  -> str:
    array = array if show_brackets else array.flatten()
    s = " ".join(str(x) for x in array)
    if encoding == 'color':
        for x in [*s]:
            try:
                s = s.replace(x, int_to_color(x))
            except:
                continue
    return s
        
class AmbigousARCDataset:
    def __init__(
        self, 
        items_data: dict = None,
        batch_size: int = 1,
        question_type: str = 'open_ended',
        example_item: bool = True,
        random_mat: bool = False,
        encoding: str = 'int',
        seed: int = None,
        concept_answer_n: int = None
    ):  
        assert question_type in ['open_ended', 'multiple_choice', 'concept_task'], f"Invalid question type '{question_type}'"
        assert encoding in ['int', 'color'], f"Invalid encoding '{encoding}'"
        if seed:    
            np.random.seed(seed)
            #random.seed(seed)
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        if items_data is None:
            from utils.globals import ITEMS_FILE
            items_data = json.load(open(ITEMS_FILE, 'rb'))

        self.items_data = items_data
        self.items = list(self.items_data.keys())[1:] # Remove example item
        self.size = len(self.items)
        self.concepts = [self.items_data[item]['concept'] for item in self.items]
        self.batch_size = batch_size
        self.num_batches = self.calculate_n_batches(batch_size)
        self.encoding = encoding
        self.random_mat = random_mat
        self.question_type = question_type

        if question_type == 'concept_task':
            self.unique_concepts = sorted(set(self.concepts))            
            self.concept_answer_n = concept_answer_n

        # Prepare the one-shot example item
        if example_item:
            if question_type == 'open_ended':
                self.example = self.open_ended('example', example=True)[0]
            else:
                if question_type == 'multiple_choice':
                    self.example = self.multiple_choice(self.items[0], random_mat=random_mat, show_answer=True)[0]  
                else: 
                    self.example = self.concept_task(self.items[0], show_answer=True)[0]

                # Remove example item from the list
                self.items = self.items[1:] 
                self.size = len(self.items)
                self.num_batches = self.calculate_n_batches(batch_size)
                self.concepts = self.concepts[1:]

        # Create the dataset
        self.x = []
        self.y = []
        for item in self.items:
            if question_type == 'open_ended':
                formatted_item, y = self.open_ended(item)  
            elif question_type == 'concept_task':
                formatted_item, y = self.concept_task(item)
            else:
                formatted_item, y = self.multiple_choice(item, random_mat=random_mat)

            if example_item:
                formatted_item = f'EXAMPLE TASK:\n\n{self.example}\n\nTEST TASK:\n\n{formatted_item}'
            self.x.append(formatted_item)
            self.y.append(y)
                                        
    def __len__(self):
        return self.num_batches
    
    def calculate_n_batches(self, batch_size):
        return self.size // batch_size + (0 if self.size % batch_size == 0 else 1)

    def __getitem__(self, idx: int):
        if idx >= self.num_batches:
            raise IndexError("Batch index out of range")
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.size)
        batch_prompts = self.x[start_idx:end_idx]
        batch_answers = self.y[start_idx:end_idx]
        return batch_prompts, batch_answers
    
    def open_ended(self, item, example=False):
        mats = ['A', 'B', 'C', 'D'] if example else ['A', 'B', 'C', 'D_matrix', 'D_concept']
        arrays = item_to_arrays(item, self.items_data, matrices=mats)
        arrays_str = [array_to_str(arr, self.encoding) for arr in arrays]
        prompt = self.mats_to_prompt(arrays_str[:3]) + '\n\nAnswer: ' 
        prompt += arrays_str[3] if example else ''
        ys = [encode_array(arr) for arr in arrays[3:]]
        return prompt, ys
    
    def multiple_choice(self, item, random_mat=False, show_answer=False):
        # Get the arrays for the item, and optionally append a random matrix
        arrays = item_to_arrays(item, self.items_data, matrices=['A', 'B', 'C', 'D_matrix', 'D_concept']) 
        if random_mat:
            rand_arr = self.random_matrix(arrays)
            arrays.append(rand_arr)   
            self.items_data[item]['D_random'] = encode_array(rand_arr)

        # Convert arrays to their string representation
        arrays_str = [array_to_str(arr, self.encoding) for arr in arrays]
        prompt = self.mats_to_prompt(arrays_str[:3]) + '\n'

        # Define answer options based on whether random_mat is used
        answer_options = ['a', 'b', 'c'] if random_mat else ['a', 'b']

        # Randomize answer order
        idx = np.random.permutation(list(range(len(answer_options)))) 
        for i in range(len(answer_options)):
            prompt += f'\n({answer_options[i]}) {arrays_str[3 + idx[i]]}'
        prompt += '\n\nAnswer: ('

        # Determine the correct answers based on the shuffled index
        matrix = answer_options[np.where(idx == 0)[0][0]]  # First matrix (correct one)
        concept = answer_options[np.where(idx == 1)[0][0]]  # Concept matrix (correct)
        random = answer_options[np.where(idx == 2)[0][0]] if random_mat else None  # Random matrix if applicable

        # Append the answer if show_answer 
        prompt += f'{concept})' if show_answer else ''

        # Return the generated prompt and a list of correct answers
        return prompt, [matrix, concept, random] if random_mat else [matrix, concept]
    
    def concept_task(self, item, show_answer=False):
        # Get the arrays for the item
        arrays = item_to_arrays(item, self.items_data, matrices=['A', 'B', 'C', 'D_concept'])
        arrays_str = [array_to_str(arr, self.encoding) for arr in arrays]
        prompt = self.mats_to_prompt(arrays_str)
        
        # Randomize the concepts
        concepts = np.random.choice(self.unique_concepts, size=len(self.unique_concepts), replace=False)
        if self.concept_answer_n:
            # Limit the number of concepts and making sure the correct concept is included
            concepts = concepts[:self.concept_answer_n]
            if self.items_data[item]['concept'] not in concepts:
                index = np.random.choice(range(self.concept_answer_n))
                concepts[-index] = self.items_data[item]['concept']

        # Capitalize and remove underscores from the concepts
        concepts_names = [' '.join(c.capitalize().split('_')) for c in concepts]
        
        # Create the answer options (a, b, c, ...)
        letters = [chr(97 + i) for i in range(len(concepts))]          

        prompt += '\n\nConcept:'
        for i in range(len(concepts_names)):
            prompt += f'\n({letters[i]}) {concepts_names[i]}'        
        prompt += f'\n\nAnswer: ('

        # Determine the correct answer
        y = letters[np.where(concepts == self.items_data[item]['concept'])[0][0]]

         # Append the answer if show_answer
        prompt += f'{y})' if show_answer else ''
        return prompt, y

    def random_matrix(self, source_mats):
        dim = source_mats[0].shape[0]
        source_mats = source_mats + [np.zeros((dim, dim), dtype=int)] # Add a zero matrix
        while True:
            random_mat = np.empty((dim, dim), dtype=int)
            for i in range(dim):
                mat = source_mats[np.random.choice([0, 1, 2])]
                random_mat[:, i] = mat[:, np.random.choice(list(range(dim)))]
                mat = source_mats[np.random.choice([0, 1, 2])]
                random_mat[i, :] = mat[np.random.choice(list(range(dim))), :]
            # Check if the random matrix is different from the source matrices
            if not any([np.array_equal(random_mat, mat) for mat in source_mats]):
                break
        return random_mat 

    def mats_to_prompt(
        self,
        matrices: List[str],
        input_str: str = '',
    ) -> str:
        '''Returns the input string for an item'''
        assert len(matrices) == 3 or len(matrices) == 4, f"Invalid number of matrices: {len(matrices)}"
        input_str += f"Input 1: " + matrices[0] + "\n"
        input_str += f"Output 1: " + matrices[1] + "\n"
        input_str += f"Input 2: " + matrices[2] + "\n"
        input_str += f"Output 2:" if len(matrices) == 3 else f"Output 2: {matrices[3]}" 
        return input_str
    
    def plot(self, item: str, title: str=None,  **kwargs):
        arrs = item_to_arrays(item, self.items_data, matrices=['A', 'B', 'C', 'D_concept', 'D_matrix', 'D_random'])
        plot_item(arrays=arrs, title=title, **kwargs)