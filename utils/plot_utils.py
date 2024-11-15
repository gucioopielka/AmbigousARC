import numpy as np
from typing import *
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, ListedColormap

def color_pallette() -> tuple:
    '''
    Creates a color palette for the plots based on the int -> color mappings.
    '''
    color_list = [color_recode(num) for num in range(8)]
    cmap = ListedColormap(color_list)
    norm = Normalize(vmin=0, vmax=7)
    return cmap, norm

def color_recode(int_encoding) -> str:
    int_to_color = {
        0: "black",
        1: "firebrick",
        2: "darkorange",
        3: "gold",
        4: "teal",
        5: "dodgerblue",
        6: "rebeccapurple",
        7: "hotpink",
    }
    return int_to_color[int_encoding]

def plot_matrix(matrix: np.array, title: str, ax=None, size=14, grid_color='grey'):
    fig, ax = plt.subplots() if ax is None else (None, ax)
    cmap, norm = color_pallette()

    # Use the created Axes object to draw the image
    cax = ax.imshow(matrix, cmap=cmap, norm=norm)
    ax.set_title(title, size=size)
    
    # Customize the axis
    midpoints = [x - 0.5 for x in range(1 + len(matrix))]
    ax.set_xticks(midpoints)
    ax.set_yticks(midpoints)
    ax.grid(True, which='both', color=grid_color, linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0)

    if ax is not None:
        return ax  
    else:
        plt.show() 
 
def plot_item(
    arrays: List[np.array],
    title: str = None,
    labels: List[str] = ['A', 'B', 'C', 'D Concept', 'D Matrix'],
    return_fig: bool = False,
    **kwargs
):  
    assert len(arrays) in [4, 5, 6], f"Invalid number of arrays: {len(arrays)}"

    # Initialize the subplots
    n_row = 2 if len(arrays) == 4 else 3
    fig, ax = plt.subplots(n_row, 3, figsize=(4, n_row+2), dpi=300)
    axs = ax.flatten()

    empty_idx = {
        4: [1, 4],
        5: [1, 4, 5, 7],
        6: [1, 4, 5]
    }.get(len(arrays))

    # Plot the matrices
    arr_idx = 0
    for i, ax in enumerate(axs):
        if i not in empty_idx:
            matrix = arrays[arr_idx]
            plot_matrix(matrix=matrix, title=labels[arr_idx], ax=ax, **kwargs)
            arr_idx += 1
        else:
            ax.axis('off')

    # Add arrows
    arrow_properties = dict(arrowstyle="->", color="black", lw=1)
    axs[0].annotate('', xy=(2, 0.5), xytext=(1.5, 0.5), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrow_properties)
    axs[3].annotate('', xy=(2, 0.5), xytext=(1.5, 0.5), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrow_properties)

    if title:
        fig.suptitle(title, size=14)

    if return_fig:
        plt.close(fig) # Close the figure to avoid displaying
        return fig, ax  
    else:
        plt.show()

arrays = [np.random.rand(5, 5) for _ in range(5)]
plot_item(arrays, title='Random Matrices', return_fig=True)