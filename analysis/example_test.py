#%% Imports
import os
import json
import pandas as pd
import numpy as np
from utils.globals import *
from utils.prompt_utils import AmbigousARCDataset
from utils.plot_utils import plot_item
from utils.eval import Eval, ModelEval
from utils.globals import ITEMS_FILE, RESULTS_DIR, get_model_name
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy.stats import ttest_ind
from rich import print as rprint

#%%
files = sorted(os.listdir(os.path.join(RESULTS_DIR, 'example_test')))
df_example = pd.DataFrame({})
for f in files:
    task = f.split('_')[0]
    concept = f.split('_')[1]
    e = Eval(os.path.join(RESULTS_DIR, 'example_test', f))
    df = e.df.groupby('model')['concept_response'].mean().sort_index().reset_index()
    df['task'] = task
    df['concept'] = concept
    df['model'] = df['model'].apply(lambda x: get_model_name(x))
    df_example = pd.concat([df_example, df])
    
# %%
df = df_example.groupby(['task', 'concept'])['concept_response'].mean().reset_index()
plt.figure(figsize=(10, 6))
for i, (task, group) in enumerate(df.groupby('task')):
    plt.subplot(1, 3, i+1)
    for j, (concept, row) in enumerate(group.iterrows()):
        plt.bar(j, row['concept_response'], color='grey')
    plt.title(task)
    plt.xticks(range(j+1), group['concept'], rotation=45)
plt.tight_layout()
plt.show()

# # %%
# tasks = df['task'].unique()
# concepts = df['concept'].unique()

# # Define bar width and create figure
# bar_width = 0.2
# plt.figure(figsize=(10, 6))

# # Set up positions for each bar based on tasks
# for i, concept in enumerate(concepts):
#     concept_data = df[df['concept'] == concept]
#     task_indices = [np.where(tasks == task)[0][0] for task in concept_data['task']]
#     plt.bar(
#         np.array(task_indices) + i * bar_width, 
#         concept_data['concept_response'], 
#         width=bar_width, 
#         label=concept
#     )

# # Set x-ticks as task names and add labels
# plt.xticks(np.arange(len(tasks)) + bar_width * (len(concepts) - 1) / 2, tasks, rotation=45)
# plt.xlabel('Task')
# plt.ylabel('Concept Response')
# plt.title('Concept Responses by Task')
# plt.legend(title='Concept')
# plt.tight_layout()
# plt.show()
# # %%

df = df_example[df_example['task'] == 'discrimination']
for model in df['model'].unique():
    model_data = df[df['model'] == model]
    plt.figure(figsize=(10, 6))
    for i, (concept, group) in enumerate(model_data.groupby('concept')):
        plt.bar(i, group['concept_response'], label=concept)
    plt.title(model)
    plt.xticks(range(i+1), model_data['concept'].unique())
    plt.legend(title='Concept')
    plt.show()


# %%


def convolve_1d(input_array, kernel_array):
    """
    Performs a 1D convolution operation (no padding, stride of 1).

    Args:
    input_array (numpy.ndarray): The input array.
    kernel_array (numpy.ndarray): The kernel array.

    Returns:
    numpy.ndarray: The result of the convolution.
    """
    # Compute the output size
    output_size = len(input_array) - len(kernel_array) + 1
    # Initialize the output array
    output_array = np.zeros(output_size)
    
    # Perform the convolution
    for i in range(output_size):
        output_array[i] = np.sum(input_array[i:i + len(kernel_array)] * kernel_array)
    
    return output_array

# Define the input and kernel arrays
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, -1])
convolve_1d(input_array, kernel_array)

# Compute the output size
output_size = len(input_array) - len(kernel_array) + 1
# Initialize the output array
output_array = np.zeros(output_size)

# Perform the convolution
for i in range(output_size):
    slice = input_array[i:i + len(kernel_array)]
    output_array[i] = np.sum(slice * kernel_array)
    np.dot(input_array[i:i + len(kernel_array)], kernel_array)

# Print the output array
print(output_array)
# %%


# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

input_matrix = np.array([[1, 2, 3], 
                         [4, 5, 6], 
                         [7, 8, 9]])
kernel_matrix = np.array([[1, 0], 
                          [0, 1]])
def compute_output_size_2d(input_matrix, kernel_matrix):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel_matrix.shape
    return (input_height - kernel_height + 1, input_width - kernel_width + 1)

compute_output_size_2d(input_matrix, kernel_matrix)

    
# %%

def convolute_2d(input_matrix, kernel_matrix):
    # Tip: same tips as above, but you might need a nested loop here in order to
    # define which parts of the input matrix need to be multiplied with the kernel matrix.
    kernel_height, kernel_width = kernel_matrix.shape
    output_height, output_width = compute_output_size_2d(input_matrix, kernel_matrix)

    output_matrix = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            # Extract the sub-matrix and compute the dot product
            sub_matrix = input_matrix[i:i + kernel_height, j:j + kernel_width]
            # dot product
            output_matrix[i, j] = np.sum(sub_matrix * kernel_matrix)
    print(output_matrix)
