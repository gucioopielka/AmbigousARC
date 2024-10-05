#import os; os.chdir('..')
import json
from utils.prompt_utils import AmbigousARCDataset
from utils.globals import *
import numpy as np
from tqdm import tqdm

randomness = []
for seed in tqdm(range(1, 1000)):
    dataset = AmbigousARCDataset(
        items_data=json.load(open(ITEMS_FILE, 'rb')),
        batch_size=1,
        task='multiple_choice',
        example_item=True,
        random_mat=False,
        seed=seed
    )

    matrix = np.abs(dataset.size/len(dataset.y[0]) - np.unique(np.array(dataset.y)[:,0], return_counts=True)[1])
    concept = np.abs(dataset.size/len(dataset.y[0])  - np.unique(np.array(dataset.y)[:,1], return_counts=True)[1])
    if dataset.random_mat:
        random = np.abs(dataset.size/len(dataset.y[0])  - np.unique(np.array(dataset.y)[:,3], return_counts=True)[1])
        randomness.append(np.average([matrix, concept, random]))
    else:
        randomness.append(np.average([matrix, concept]))

print(list(range(1, 1000))[np.argmin(randomness)])

dataset = AmbigousARCDataset(
    items_data=json.load(open(ITEMS_FILE, 'rb')),
    batch_size=1,
    task='multiple_choice',
    example_item=True,
    random_mat=False,
    seed=42
)

matrix = np.abs(dataset.size/len(dataset.y[0]) - np.unique(np.array(dataset.y)[:,0], return_counts=True)[1])
concept = np.abs(dataset.size/len(dataset.y[0])  - np.unique(np.array(dataset.y)[:,1], return_counts=True)[1])
print(matrix, concept)



seeds = []
for seed in tqdm(range(1, 10000)):
    dataset = AmbigousARCDataset(
        items_data=json.load(open(ITEMS_FILE, 'rb')),
        batch_size=1,
        task='concept_task',
        example_item=True,
        random_mat=False,
        seed=9123,
        concept_answer_n=13
    )

    seeds.append(np.mean(np.abs(dataset.size/len(set(dataset.y)) - np.unique(dataset.y, return_counts=True)[1])))

print(list(range(1, 10000))[np.argmin(seeds)], ' ',np.unique(dataset.y, return_counts=True)[1])

seeds[np.argmax(seeds)]
seeds == np.min(seeds)