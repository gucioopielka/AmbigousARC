import numpy as np
import json
import hashlib
from typing import *
from utils.plot_utils import *
from scipy.stats import chisquare
from itertools import permutations
from tqdm import tqdm

def convert_to_array(x: str, dim: int) -> np.array:
    return np.array([int(char) for char in x]).reshape(dim, dim)

def encode_array(array: np.array) -> str:
    return ''.join([str(x) for x in array.flatten()])

def item_to_arrays(items_data: dict, matrices: List[str] = ['A', 'B', 'C']) -> List[np.array]:    
    # Check for each matrix in the item's data
    missing_matrices = [mat for mat in matrices if mat not in items_data]
    if missing_matrices:
        raise ValueError(f"Matrix (or matrices) {missing_matrices} not found in item")
    return [convert_to_array(items_data[mat], items_data['xdim']) for mat in matrices]

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

def generate_seed(value, seed=None):
    hash_object = hashlib.md5(value.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    return (hash_int + (seed if seed else 0)) % (2**32 - 1)

def create_mirror_data(data: list, n_mirror: int, seed: int=None, max_iter=10):
    # All possible matrices
    matrices = ['A', 'B', 'C', 'D_Concept', 'D_Matrix', 'D_random']

    def create_mirror_color_map(item: dict, seed=None) -> dict:
        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility

        # Get all the colors (using a sorted list for determinism)
        all_colors = sorted([str(i) for i in range(1, 8)])  # Sort to ensure deterministic order

        # Get unique colors in the item
        mats = ''
        for k, v in item.items(): 
            if k in matrices:
                mats += v

        item_colors = sorted(set([c for c in mats if c != '0']))  # Sort for determinism after set creation

        # Create a new color mapping
        new_colors = {color: None for color in item_colors}

        for color in item_colors:
            # Get the possible colors: all colors - current color - already used colors
            used_colors = set(new_colors.values()) - {None}  # Ensure None is excluded
            possible_colors = sorted(set(all_colors) - {color} - used_colors)  # Sort to ensure deterministic selection
            new_colors[color] = np.random.choice(possible_colors)

        new_colors['0'] = '0'  # Black is always black

        return new_colors

    def assign_color_map(item: dict, color_map: dict) -> dict:
        new_item = {}
        for k, v in item.items():
            if k in matrices:
                new_item[k] = ''.join([color_map[c] for c in v])
            else:
                new_item[k] = v
        return new_item
    
    new_data = []
    for item in data:
        # Change the id of the original item
        original_item = item.copy()
        original_item['id'] = original_item["id"] + '_1'
        new_data.append(original_item)
    
        # Create mirror color maps
        color_maps = []
        for mirror_index in range(n_mirror):
            i = 0
            while True:
                if seed:
                    # Generate a consistent seed for each mirror item
                    current_seed = seed + mirror_index + i
                color_map = create_mirror_color_map(item, current_seed)
                # Check if the color map is unique
                if color_map not in color_maps:
                    break
                else:
                    i += 1 
                    if i > max_iter:
                        raise ValueError("Could not create a unique color map")
            color_maps.append(color_map)
            
        # Create mirror items
        for idx, color_map in enumerate(color_maps):
            new_item = assign_color_map(item, color_map)
            new_item['id'] = item["id"].split('_')[0] + '_' + str(idx+2)
            new_data.append(new_item)
    return new_data

def find_seed(data_cfg, n_iter = 100):
    '''
    Find the least biased seed for the dataset by comparing the observed frequencies of the answer options
    with the expected frequencies of the answer options. The expected frequencies are generated by all possible
    integer combinations that sum to the dataset size.

    The function stops when the observed frequencies match one of the expected integer combinations.
    '''
    print("Finding the best seed for the dataset...")

    def generate_expected_combinations(total, num_categories):
        """
        Generate all possible combinations of integers that sum to 'total'
        across 'num_categories' categories.
        """
        # Start with the base expected value (floor division)
        base = total // num_categories
        remainder = total % num_categories
        
        # Create the base array
        base_expected = np.array([base] * num_categories)
        
        # Create variations by distributing the remainder across categories
        for indices in permutations(range(num_categories), remainder):
            variation = base_expected.copy()
            for idx in indices:
                variation[idx] += 1
            yield variation

    data_cfg['find_best_seed'] = False
    expected_combinations = None
    randomness = {
        'seed': [],
        'p_value': [],
        'observed': [],
    }
    for seed in tqdm(range(1, n_iter)):
        data_cfg['seed'] = seed
        dataset = AmbigousARCDataset(**data_cfg)              
        y = np.array(dataset.y)
        y = y[:, 0] if len(y.shape) > 1 else y
        observed = np.unique(y, return_counts=True)[1]

        # Generate all possible expected integer combinations that sum to dataset.size
        if expected_combinations is None:
            expected_combinations = list(generate_expected_combinations(dataset.size, len(dataset.answer_options)))
        
        # Continue storing values for randomness analysis
        randomness['seed'].append(seed)
        randomness['observed'].append(observed)

        # Compute the chi-square test only when observed length matches the expected
        if len(observed) == len(expected_combinations[0]):
            randomness['p_value'].append(chisquare(f_obs=observed, f_exp=expected_combinations[0])[1])
        else:
            randomness['p_value'].append(None)

        # Check if the observed matches any of the expected combinations
        if any(np.array_equal(observed, exp) for exp in expected_combinations):
            print(f"Stopping at seed {seed}: observed matches one of the expected integer combinations.")
            break

    best_seed = randomness['seed'][np.argmax(randomness['p_value'])]
    observed_best = randomness['observed'][np.argmax(randomness['p_value'])]
    observed_best = {chr(97 + i): observed_best[i] for i in range(len(observed_best))}
    print(f"Best seed: {best_seed} || Observed: {observed_best} || p-value: {round(max(randomness['p_value']), 2)}")
    return best_seed

class AmbigousARCDataset:
    def __init__(
        self, 
        items_data: List|str = None,
        batch_size: int = 1,
        task: str = 'generation',
        example_item: bool = True,
        random_mat: bool = True,
        duplicate: str = None,
        encoding: str = 'int',
        find_best_seed: bool = False,
        n_iter_seed: int = 100,
        seed: int = None,
        seed_static = 42,
        concept_answer_n: int = 3,
        n_mirror: int = 0
    ):  
        assert task in ['generation', 'discrimination', 'recognition'], f'Invalid task "{task}"'
        assert encoding in ['int', 'color'], f"Invalid encoding type '{encoding}'"
        assert duplicate in [None, 'random', 'a', 'b', 'c'], f"Invalid duplicate type '{duplicate}'"

        # Load the items data
        if items_data is None:
            from utils.globals import ITEMS_FILE
            items_data = json.load(open(ITEMS_FILE, 'rb'))
            self.items_data_path = str(ITEMS_FILE)
        elif isinstance(items_data, str):
            self.items_data_path = items_data
            items_data = json.load(open(items_data, 'rb'))

        # Set the configuration
        self.encoding = encoding
        self.batch_size = batch_size
        self.random_mat = random_mat
        self.duplicate = duplicate
        self.task = task
        self.seed = seed
        self.seed_static = seed_static
        self.concept_answer_n = concept_answer_n
        self.n_mirror = n_mirror
        self.example_item = example_item
        
        # Find the best seed for the dataset
        if find_best_seed:
            assert self.task in ['discrimination', 'recognition'], f"Only 'discrimination' and 'recognition' tasks are supported for finding the best seed."
            cfg = {'items_data': self.items_data_path, **self.get_config()}
            self.seed = find_seed(cfg, n_iter_seed)

        # Generate random matrices for the items
        if random_mat and task == 'discrimination':
            for idx, item in enumerate(items_data):
                items_data[idx]['D_random'] = encode_array(self.random_matrix(item, seed_static))

        # Create mirror items
        if n_mirror > 0:
            items_data = create_mirror_data(items_data, n_mirror, seed_static)
        else:
            # _1 signifies the original item
            for item in items_data:
                item['id'] = item['id'] + '_1'

        # Set the dataset properties
        self.items_data = items_data
        self.item_ids = [item['id'] for item in self.items_data]
        self.size = len(self.items_data) # Number of all items (including mirrors)
        self.n_items = len(set(i.split('_')[0] for i in self.item_ids)) # Main items
        self.concepts = [item['concept'] for item in self.items_data]
        self.unique_concepts = sorted(set(self.concepts))            
        self.num_batches = self.calculate_n_batches(batch_size)
        self.concept_answer_n = len(self.unique_concepts) if concept_answer_n == -1 else concept_answer_n
        self.input_matrices = [[item['A'], item['B'], item['C']] for item in self.items_data]
        self.__concepts_to_remove = []

        # Create the dataset
        self.x = []
        self.y = []
        self.example_items = [] if example_item else None
        self.example_y = [] if example_item else None
        for item in self.items_data:
            # Update the seed for main items (mirrors are generated based on the main item)
            if item['id'].split('_')[1] == '1':
                self.__seed_item = generate_seed(item['id'], self.seed)
                seed_static_item = generate_seed(item['id'], seed_static)

            # Set the seed for the item (otherwise it biases the answer options)
            np.random.seed(self.__seed_item)
            
            if example_item:
                # Get an example item that comes from a different concept
                rng_static = np.random.default_rng(seed_static_item) # For consistency across tasks
                other_items = [i for i in self.items_data if i['concept'] != item['concept']]
                example_item_data = rng_static.choice(other_items)

                # Create the example item
                example_prompt, example_y = getattr(self, task)(example_item_data, example=True)
                self.example_items.append(example_item_data['id'])
                self.example_y.append(example_y)
            
            # Generate the item
            formatted_item, y = getattr(self, task)(item, example=False)

            # Add the example to the item
            formatted_item = f'EXAMPLE TASK:\n\n{example_prompt}\n\nTEST TASK:\n\n{formatted_item}' if example_item else formatted_item
            self.x.append(formatted_item)
            self.y.append(y)
                                        
    def __len__(self):
        return self.num_batches
    
    def calculate_n_batches(self, batch_size: int) -> int:
        return self.size // batch_size + (0 if self.size % batch_size == 0 else 1)

    def __getitem__(self, idx: int):
        if idx >= self.num_batches:
            raise IndexError("Batch index out of range")
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.size)
        batch_prompts = self.x[start_idx:end_idx]
        batch_answers = self.y[start_idx:end_idx]
        return batch_prompts, batch_answers
    
    def generation(self, item: dict, example: bool=False) -> Tuple[str, List[str]]:
        mats = ['A', 'B', 'C', 'D_Concept'] if example else ['A', 'B', 'C', 'D_Matrix', 'D_Concept']
        arrays = item_to_arrays(item, matrices=mats)
        arrays_str = [array_to_str(arr, self.encoding) for arr in arrays]
        prompt = self.mats_to_prompt(arrays_str[:3]) + '\n\nAnswer: ' 
        prompt += arrays_str[3] if example else ''
        ys = [encode_array(arr) for arr in arrays[3:]]
        return prompt, ys
    
    def discrimination(self, item: dict, example: bool=False) -> Tuple[str, List[str]]:
        # Get the arrays for the item
        matrices = ['A', 'B', 'C', 'D_Matrix', 'D_Concept']
        matrices += ['D_random'] if self.random_mat else []
        arrays = item_to_arrays(item, matrices=matrices)  
        
        # Create a duplicate answer option
        if self.duplicate == 'random':
            # Randomly duplicate one of the input matrices
            arrays.append(arrays[np.random.choice([0, 1, 2])]) 
        elif self.duplicate in ['a', 'b', 'c']:
            # Duplicate the specified matrix
            arrays.append(arrays['abc'.index(self.duplicate)]) 

        # Convert arrays to their string representation
        arrays_str = [array_to_str(arr, self.encoding) for arr in arrays]
        prompt = self.mats_to_prompt(arrays_str[:3]) + '\n'

        # Randomize answer order
        self.answer_options = [chr(97 + i) for i in range(len(arrays_str) - 3)]
        idx = np.random.permutation(list(range(len(self.answer_options)))) 
        for i in range(len(self.answer_options)):
            prompt += f'\n({self.answer_options[i]}) {arrays_str[3 + idx[i]]}'
        prompt += '\n\nAnswer: ('

        # Determine the correct answers based on the shuffled index
        matrix = self.answer_options[np.where(idx == 0)[0][0]]  # Matrix
        concept = self.answer_options[np.where(idx == 1)[0][0]]  # Concept matrix (correct)
        random = self.answer_options[np.where(idx == 2)[0][0]] if self.random_mat else None  # Random matrix if applicable
        duplicate = self.answer_options[np.where(idx == 3)[0][0]] if self.duplicate else None  # Duplicate matrix if applicable

        # Append the answer if show_answer 
        prompt += f'{concept})' if example else ''

        # Return the generated prompt and a list of correct answers
        return prompt, [matrix, concept, random, duplicate]
    
    def recognition(self, item: dict, example: bool=False) -> Tuple[str, str]:
        # Convert the item's arrays to prompt format
        arrays = item_to_arrays(item, matrices=['A', 'B', 'C', 'D_Concept'])
        arrays_str = [array_to_str(arr, self.encoding) for arr in arrays]
        prompt = self.mats_to_prompt(arrays_str)
        
        if not example:
            # Remove the concepts that were used in the example item
            concepts_choices = [c for c in self.unique_concepts if c not in self.__concepts_to_remove]
        else:
            concepts_choices = self.unique_concepts

        # Randomize the concepts
        concepts = np.random.choice(concepts_choices, size=len(concepts_choices), replace=False)
        
        # Limit the number of concepts 
        concepts = concepts[:self.concept_answer_n]
        concepts = concepts.astype('<U50')

        # making sure the correct concept is included
        if item['concept'] not in concepts:
            index = np.random.choice(range(len(concepts)))
            concepts[index] = item['concept']

        # Capitalize and remove underscores from the concepts
        concepts_names = [' '.join(c.capitalize().split('_')) for c in concepts]
        
        # Create the answer options (a, b, c, ...)
        self.answer_options = [chr(97 + i) for i in range(len(concepts))]          
        prompt += '\n\nConcept:'
        for i in range(len(concepts_names)):
            prompt += f'\n({self.answer_options[i]}) {concepts_names[i]}'        
        prompt += f'\n\nAnswer: ('

        # Determine the correct answer
        y = self.answer_options[np.where(concepts == item['concept'])[0][0]]

         # Append the answer if show_answer
        prompt += f'{y})' if example else ''

        # Save the concepts to remove for the actual item
        if example:
            self.__concepts_to_remove = concepts

        return prompt, y

    def random_matrix(self, item: dict, seed: int=None) -> np.array:      
        if seed:
            np.random.seed(seed)

        # Get the source matrices
        source_mats = item_to_arrays(item, matrices=['A', 'B', 'C']) 
        dim = source_mats[0].shape[0]

        # The random matrix is created by randomly selecting columns and rows from the source matrices
        while True:
            random_mat = np.empty((dim, dim), dtype=int)
            for i in range(dim):
                mat = source_mats[np.random.choice([0, 1, 2])]
                random_mat[:, i] = mat[:, np.random.choice(list(range(dim)))]
                mat = source_mats[np.random.choice([0, 1, 2])]
                random_mat[i, :] = mat[np.random.choice(list(range(dim))), :]
            
            # Check if the random matrix is different from the source matrices
            if not any([np.array_equal(random_mat, mat) for mat in source_mats]):
                return random_mat

    def mats_to_prompt(self, matrices: List[str], input_str: str = '') -> str:
        '''Returns the input string for an item'''
        assert len(matrices) == 3 or len(matrices) == 4, f"Invalid number of matrices: {len(matrices)}"
        input_str += f"Input 1: " + matrices[0] + "\n"
        input_str += f"Output 1: " + matrices[1] + "\n"
        input_str += f"Input 2: " + matrices[2] + "\n"
        input_str += f"Output 2:" if len(matrices) == 3 else f"Output 2: {matrices[3]}" 
        return input_str

    def plot(self, item: int|str, title: str=None, show_mirrors: bool=False,  **kwargs):
        if isinstance(item, int):
            item_id = self.item_ids[item].split('_')[0]
        if isinstance(item, str):
            item_id = item.split('_')[0] if '_' in item else item
        
        # Get the item and its mirrors
        items = [i for i in self.items_data if i['id'].startswith(item_id)]
        items = items if show_mirrors else [items[0]] # Only show the first item

        # Define the matrices to plot based on the task
        mats = ['A', 'B', 'C', 'D_Concept']
        if self.task == 'generation':
            mats += ['D_Matrix']
        if self.task == 'discrimination':
            mats += ['D_Matrix', 'D_random']

        for item in items:
            arrs = item_to_arrays(item, matrices=mats)
            labels = mats[:3] + ['D'] if self.task == 'recognition' else mats
            plot_item(arrays=arrs, title=title, labels=labels, **kwargs)
    
    def get_config(self) -> dict:
        return {
            'task': self.task,
            'example_item': self.example_item,
            'encoding': self.encoding,
            'random_mat': self.random_mat,
            'duplicate': self.duplicate,
            'concept_answer_n': self.concept_answer_n,
            'seed': self.seed,
            'seed_static': self.seed_static,
            'batch_size': self.batch_size,
            'n_mirror': self.n_mirror,
            'example_item': self.example_item
        }