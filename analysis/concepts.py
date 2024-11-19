#%% Imports
import os
import json
import pandas as pd
import numpy as np
from utils.globals import *
from utils.prompt_utils import AmbigousARCDataset
from utils.plot_utils import plot_item, get_percentage_ticks
from utils.eval import Eval, ModelEval
from utils.globals import ITEMS_FILE, RESULTS_DIR, get_model_name
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

run_dir = 'run_2024-11-12' 
generation = Eval(os.path.join(RESULTS_DIR, run_dir, 'generation.json'))
discrimination = Eval(os.path.join(RESULTS_DIR, run_dir, 'discrimination.json'))
recognition = Eval(os.path.join(RESULTS_DIR, run_dir, 'recognition.json'))

# %%
for idx, task in enumerate(['generation', 'discrimination']):
    response_cols = ['concept_response', 'matrix_response', 'duplicate_response']

    means = eval(task).df.groupby('concept')[response_cols].mean()
    sems = eval(task).df.groupby(['concept', 'model'])[response_cols].sem().groupby('concept').mean()

    # Sort the DataFrame
    if idx == 0:
        means = means.sort_values(by='concept_response', ascending=True)
        index = means.index
    else:
        means = means.loc[index]

    sems = sems.loc[means.index]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 20))
    means.plot(
        kind='barh', 
        ax=ax, 
        color=['#d44e86', '#665190', '#adf4dc', '#f4ad42'], 
        edgecolor='black', 
        xerr=sems, 
        #capsize=5
    )
    plt.ylabel('')
    plt.xticks(*get_percentage_ticks(means), size=18)
    plt.yticks(size=18)
    plt.xlabel('% of Responses', size=18)
    plt.title(f'{task.capitalize()} Task', size=20)
    plt.show()

# %%
