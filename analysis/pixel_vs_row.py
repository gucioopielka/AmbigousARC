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
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy.stats import ttest_ind
from rich import print as rprint

def plot_pixel_vs_row(df_pixel, df_row, title=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    bar_width = 0.2
    x = np.arange(len(df_pixel))
    colors = ['#B53A3D', '#508FA3', '#F3E4D0', '#E9C06C']

    # Plot Pixel - Concept and Matrix bars
    ax.bar(x - bar_width * 1.5, df_pixel['concept_response'], bar_width, label="Pixel - Concept", color=colors[0], edgecolor='grey', alpha=0.9)
    ax.bar(x - bar_width * 0.5, df_row['concept_response'], bar_width, label="Row - Concept", color=colors[0], edgecolor='black', hatch='//')

    # Plot Row - Concept and Matrix bars
    ax.bar(x + bar_width * 0.5, df_pixel['matrix_response'], bar_width, label="Pixel - Matrix", color=colors[1], edgecolor='grey', alpha=0.9)
    ax.bar(x + bar_width * 1.5, df_row['matrix_response'], bar_width, label="Row - Matrix", color=colors[1], edgecolor='black', hatch='//')

    # Customize plot appearance
    ax.set_xticks(x)
    ax.set_xticklabels([get_model_name(i) for i in df_pixel.index], rotation=45, ha='right', fontsize=16)
    ax.set_ylabel('% of responses', fontsize=16)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels(['{:.0f}%'.format(i * 100) for i in np.arange(0, 1.1, 0.2)], fontsize=16)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)

    legend_pixel = Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', label='Pixel')
    legend_row = Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', label='Row', hatch='///')
    legend_entries = [legend_pixel, legend_row] + [
        Rectangle((0, 0), 1, 1, color=color, edgecolor='black', label=col) for color, col in zip(colors, ['Concept', 'Matrix'])
    ]

    # Add custom legend
    ax.legend(handles=legend_entries, frameon=False, fontsize=12, bbox_to_anchor=(1, 1))

    # Add title
    if title:
        plt.title(title, fontsize=18)

    plt.tight_layout()
    plt.show()

def compare_means(df_1: pd.DataFrame, df_2: pd.DataFrame):
    pixel_mean, pixel_std = df_1.groupby('model')['matrix_response'].mean().describe()[['mean', 'std']].values
    row_mean, row_std = df_2.groupby('model')['matrix_response'].mean().describe()[['mean', 'std']].values
    p_value = ttest_ind(df_1.groupby('model')['matrix_response'].mean(), df_2.groupby('model')['matrix_response'].mean()).pvalue
    return (pixel_mean, pixel_std), (row_mean, row_std), p_value


#%% Discrimination
file_pixel = os.path.join(RESULTS_DIR, 'pixel_vs_row/pixel.json')
file_row = os.path.join(RESULTS_DIR, 'pixel_vs_row/row.json')
file_generation = os.path.join(RESULTS_DIR, 'run_2024-11-12/generation.json')
pixel = Eval(file_pixel)
row = Eval(file_row)
 
# Means 
(pixel_mean, pixel_std), (row_mean, row_std), p_value = compare_means(pixel.df, row.df)
rprint(
    f"[u][i]Proportion Matrix Responses[/i][/u] (N = {len(pixel.df['model'].unique())})\n",
    '\nPixel:', 
    f'[yellow][u]{pixel_mean:.2f}[/u][/yellow]', f'({pixel_std:.2f})', 
    '\nRow:', 
    f'[yellow][u]{row_mean:.2f}[/u][/yellow]', f'({row_std:.2f})',
    '\n\nP-value:', 
    f'[green]{p_value:.3f}[/green]' if p_value < 0.05 else f'[red]{p_value:.3f}[/red]', 
)

# Plot the data
response_cols = ['concept_response', 'matrix_response']
df_pixel = pixel.df.groupby('model')[response_cols].mean()
df_pixel.sort_values('concept_response', ascending=True, inplace=True)   
df_row = row.df.groupby('model')[response_cols].mean()
df_row = df_row.loc[df_pixel.index]
plot_pixel_vs_row(df_pixel, df_row, 'Discrimination')


#%% Plot the items
dataset_pixel = AmbigousARCDataset(d_matrix_level='pixel')
dataset_row = AmbigousARCDataset(d_matrix_level='row')

item_df = pixel.df.groupby('item_main_id')['matrix_response'].mean().reset_index()
item_df.rename(columns={'matrix_response': 'matrix_response_pixel'}, inplace=True)
item_df['matrix_response_row'] = row.df.groupby('item_main_id')['matrix_response'].mean().values
item_df = item_df.sort_values('matrix_response_row', ascending=False).reset_index(drop=True)

main_fig, main_axs = plt.subplots(len(item_df['item_main_id']), 2, figsize=(8, 30))
for idx, item in enumerate(item_df['item_main_id']):
    fig_pixel, _ = dataset_pixel.plot(item, return_fig=True)
    fig_pixel.canvas.draw()
    fig_row, _ = dataset_row.plot(item, return_fig=True)
    fig_row.canvas.draw()

    main_axs[idx, 0].imshow(np.asarray(fig_pixel.canvas.buffer_rgba()))
    main_axs[idx, 0].set_title(f'Pixel - {item_df.loc[idx, "matrix_response_pixel"].round(2) * 100:.0f}%')
    main_axs[idx, 1].imshow(np.asarray(fig_row.canvas.buffer_rgba()))
    main_axs[idx, 1].set_title(f'Row - {item_df.loc[idx, "matrix_response_row"].round(2) * 100:.0f}%')
    
    main_axs[idx, 0].axis('off')
    main_axs[idx, 1].axis('off')
plt.tight_layout()
plt.show()


#%% Generation 
above_below_items = ["0Zgi5T", "3qC5EW", "DAyZ1w", "MXPEB2", "NCz8AP", "TKKBpo", "ckjZ81", "hC2gHL", "zYuq0D"]
response_cols = ['concept_response', 'matrix_response', 'duplicate_response']
dataset_cfg = json.load(open(file_generation, 'rb'))['dataset_config']

# Pixel
dataset_cfg['d_matrix_level'] = 'pixel'
dataset = AmbigousARCDataset(**dataset_cfg)
generation_pixel = Eval(file_generation, dataset=dataset)
gen_pixel_df = generation_pixel.df[generation_pixel.df['item_main_id'].isin(above_below_items)]
assert gen_pixel_df.shape[0] == pixel.df.shape[0], 'Different number of items in the generation and discrimination datasets'

# Row
dataset_cfg['d_matrix_level'] = 'row'
dataset = AmbigousARCDataset(**dataset_cfg)
generation_row = Eval(file_generation, dataset=dataset)
gen_row_df = generation_row.df[generation_row.df['item_main_id'].isin(above_below_items)]
assert gen_row_df.shape[0] == row.df.shape[0], 'Different number of items in the generation and discrimination datasets'

# Means
(pixel_mean, pixel_std), (row_mean, row_std), p_value = compare_means(gen_pixel_df, gen_row_df)
rprint(
    f"[u][i]Proportion Matrix Responses[/i][/u] (N = {len(gen_pixel_df['model'].unique())})\n",
    '\nPixel:', 
    f'[yellow][u]{pixel_mean:.2f}[/u][/yellow]', f'({pixel_std:.2f})', 
    '\nRow:', 
    f'[yellow][u]{row_mean:.2f}[/u][/yellow]', f'({row_std:.2f})',
    '\n\nP-value:', 
    f'[green]{p_value:.3f}[/green]' if p_value < 0.05 else f'[red]{p_value:.3f}[/red]', 
)

# Plot the data
response_cols = ['concept_response', 'matrix_response']
df_pixel = gen_pixel_df.groupby('model')[response_cols].mean()
df_row = gen_row_df.groupby('model')[response_cols].mean()
plot_pixel_vs_row(df_pixel, df_row, 'Generation')

#%% Rotated items
response = 'matrix_response'

# Original
row_ms = [row.df[response].mean()]
row_sems = [row.df.groupby('model')[response].mean().sem()]
pixel_ms = [pixel.df[response].mean()]
pixel_sems = [pixel.df.groupby('model')[response].mean().sem()]

# Rotated
for angle in [90, 180, 270]:
    file_row_rotated = os.path.join(RESULTS_DIR, f'pixel_vs_row/row_rotated_{angle}.json')
    file_pixel_rotated = os.path.join(RESULTS_DIR, f'pixel_vs_row/pixel_rotated_{angle}.json')
    row_rotated = Eval(file_row_rotated)
    pixel_rotated = Eval(file_pixel_rotated)

    row_ms.append(row_rotated.df[response].mean())
    row_sems.append(row_rotated.df.groupby('model')[response].mean().sem())
    pixel_ms.append(pixel_rotated.df[response].mean())
    pixel_sems.append(pixel_rotated.df.groupby('model')[response].mean().sem())

plt.figure(figsize=(4.5, 5))
plt.errorbar([0, 90, 180, 270], row_ms, yerr=row_sems, label='Row', color='blue', marker='o', capsize=5)
plt.errorbar([0, 90, 180, 270], pixel_ms, yerr=pixel_sems, label='Pixel', color='red', marker='o', capsize=5)
plt.xticks([0, 90, 180, 270], ['0', '90', '180', '270'])
plt.ylabel('Matrix Response', fontsize=12)
plt.ylim(0, 1)
plt.legend(frameon=False)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.tight_layout()
plt.show()

#%% Plot the items
dataset_row_original = AmbigousARCDataset(d_matrix_level='row', matrix_rotation=False)
dataset_row_rotated = AmbigousARCDataset(d_matrix_level='row', matrix_rotation=True)

item_df = row.df.groupby('item_main_id')['matrix_response'].mean().reset_index()
item_df.rename(columns={'matrix_response': 'matrix_response_row_original'}, inplace=True)
item_df['matrix_response_row_rotated'] = row_rotated.df.groupby('item_main_id')['matrix_response'].mean().values
item_df = item_df.sort_values(['matrix_response_row_rotated', 'matrix_response_row_original'], ascending=False).reset_index(drop=True)

main_fig, main_axs = plt.subplots(len(item_df['item_main_id']), 2, figsize=(8, 30))
for idx, item in enumerate(item_df['item_main_id']):
    fig_pixel, _ = dataset_row_original.plot(item, return_fig=True)
    fig_pixel.canvas.draw()
    fig_row, _ = dataset_row_rotated.plot(item, return_fig=True)
    fig_row.canvas.draw()

    main_axs[idx, 0].imshow(np.asarray(fig_pixel.canvas.buffer_rgba()))
    main_axs[idx, 0].set_title(f'Original - {item_df.loc[idx, "matrix_response_row_original"].round(2) * 100:.0f}%')
    main_axs[idx, 1].imshow(np.asarray(fig_row.canvas.buffer_rgba()))
    main_axs[idx, 1].set_title(f'Rotated - {item_df.loc[idx, "matrix_response_row_rotated"].round(2) * 100:.0f}%')
    
    main_axs[idx, 0].axis('off')
    main_axs[idx, 1].axis('off')
plt.tight_layout()
plt.show()

# %%
print(dataset_row_original.x[0])
dataset_row_original.plot(0)
print(dataset_row_rotated.x[0])

dataset_row_rotated.plot(0)

# %%
def plot_item_responses(item_id):
    row.df[row.df['item_main_id'] == item_id].groupby('model')['concept_response'].mean().plot(kind='bar', figsize=(10, 5))
plot_item_responses('MXPEB2')