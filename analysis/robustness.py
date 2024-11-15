import os
import json
import pandas as pd
import numpy as np
from utils.globals import *
from utils.plot_utils import plot_item
from utils.eval import Eval, ModelEval
from utils.globals import ITEMS_FILE, RESULTS_DIR, MODEL_NAMES, get_model_name
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

run_dir = 'run_2024-11-12'
generation_color = Eval(os.path.join(RESULTS_DIR, run_dir, 'generation.json'))
discrimination_color = Eval(os.path.join(RESULTS_DIR, run_dir, 'discrimination.json'))
recognition_color = Eval(os.path.join(RESULTS_DIR, run_dir, 'recognition.json'))

run_dir = 'run_example_2024-11-12'
generation_example = Eval(os.path.join(RESULTS_DIR, run_dir, 'generation.json'))
discrimination_example = Eval(os.path.join(RESULTS_DIR, run_dir, 'discrimination.json'))
recognition_example = Eval(os.path.join(RESULTS_DIR, run_dir, 'recognition.json'))

# Combine the dataframes
df = pd.DataFrame({})
for task in ['generation', 'discrimination', 'recognition']:
    # Create DataFrame
    df_color = pd.DataFrame(eval(f'{task}_color').df.groupby(['model', 'mirror'])['concept_response'].mean().reset_index())
    df_example = pd.DataFrame(eval(f'{task}_example').df.groupby(['model', 'mirror'])['concept_response'].mean().reset_index())
    df_color['example_acc'] = df_example['concept_response']
    df_task = df_color.rename(columns={'concept_response': 'color_acc'})
    df_task['task'] = task
    df_task['model'] = df_task['model'].apply(lambda x: get_model_name(x))
    df = pd.concat([df, df_task])

# Plotting
offset = 0.13
for task in ['generation', 'discrimination', 'recognition']:
    df_task = df[df['task'] == task]
    means = df_task.groupby('model')[['example_acc']].mean().sort_values('example_acc', ascending=True)
    df_task['model'] = pd.Categorical(df_task['model'], categories=means.index, ordered=True)
    df_task = df_task.sort_values('model')

    plt.figure(figsize=(8, 6))
    for i, (model, group) in enumerate(df_task.groupby('model')):
        # Scatter plot for 'example_acc' with a slight offset to the left
        plt.scatter(
            [i - offset] * len(group),
            group['example_acc'],
            label='example_acc',
            s=100,
            alpha=0.5,
            color='grey'
        )
        
        # Scatter plot for 'color_acc' with a slight offset to the right
        plt.scatter(
            [i + offset] * len(group),
            group['color_acc'],
            label='color_acc',
            s=100,
            alpha=0.5,
            color='blue'
        )
    plt.xticks([i for i in range(len(MODEL_NAMES))], [get_model_name(i) for i in df_task.groupby('model')['model'].first().values], rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('Concept Response', fontsize=16)
    plt.tight_layout()
    plt.title(f'{task.capitalize()} Task', fontsize=20)
    #plt.legend()
    plt.show()