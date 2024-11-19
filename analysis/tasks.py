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
mpl.rcParams['figure.dpi'] = 300

run_dir = 'run_2024-11-12' 
generation = Eval(os.path.join(RESULTS_DIR, run_dir, 'generation.json'))
discrimination = Eval(os.path.join(RESULTS_DIR, run_dir, 'discrimination.json'))
recognition = Eval(os.path.join(RESULTS_DIR, run_dir, 'recognition.json'))

# %% Task Performance
df = pd.DataFrame({
    'Discrimination': discrimination.df.groupby('model')['concept_response'].mean(),
    'Recognition': recognition.df.groupby('model')['concept_response'].mean(),
    'Generation': generation.df.groupby('model')['concept_response'].mean(),
})

# Sort the DataFrame
df = df.sort_values(by=['Discrimination', 'Recognition'], ascending=True)# Sort the DataFrame

# Model names adjustment
model_names = [get_model_name(name) for name in df.index]

# Plotting
ax = df.plot(kind='bar', figsize=(9, 6), color=['#d44e86', '#665190', '#adf4dc'], edgecolor='black')

# Set axis labels and ticks
plt.ylabel('Accuracy', size=16)
plt.xticks(np.arange(0, len(df), 1), model_names, size=14, rotation=45, ha='right')
plt.yticks(np.linspace(0, 0.8, num=5), [f'{x:.0%}' for x in np.linspace(0, 0.8, num=5)], size=14)
plt.xlabel('')
plt.ylim(0, 0.8)

# Calculate overall means
mean_discrimination = df['Discrimination'].mean()
mean_recognition = df['Recognition'].mean()
mean_generation = df['Generation'].mean()

# Overlay circles on the y-axis for the overall means
plt.scatter(-0.5, mean_discrimination, color='#d44e86', s=300, zorder=0, edgecolor='black', marker='>')
plt.scatter(-0.5, mean_recognition, color='#665190', s=350, zorder=0, edgecolor='black', marker='>')
plt.scatter(-0.5, mean_generation, color='#adf4dc', s=350, zorder=0, edgecolor='black', marker='>')

# Customize the legend
plt.legend(fontsize=14, edgecolor='none')

# Remove unnecessary spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.axhline(y=1/4, color='grey', linestyle='--', linewidth=1, zorder=0)

# Show the plot
plt.tight_layout()
plt.show()


# Correlation between tasks
df_con = pd.DataFrame({
    'Discrimination': discrimination.df.groupby('model')['duplicate_response'].mean(),
    #'Recognition': recognition.df.groupby('model')['matrix_response'].mean(),
    'Generation': generation.df.groupby('model')['duplicate_response'].mean(),
})
df_con.corr().round(2)   


#%% Response Types
response_cols = ['concept_response', 'matrix_response', 'duplicate_response']
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
df = generation.df.groupby('model')[response_cols].mean().sort_values('concept_response', ascending=True)
df.plot(kind='bar', stacked=False, color=['#B53A3D', '#508FA3', '#F3E4D0'], ax=ax[0], edgecolor='black')
ax[0].set_title('Generation', fontsize=16)
ax[0].set_ylabel('% of responses', fontsize=14)
ax[0].set_yticks(np.arange(0, 0.5, 0.1), ['{:.0f}%'.format(i*100) for i in np.arange(0, 0.5, 0.1)], fontsize=12)
ax[0].set_xticks(range(df.shape[0]), [get_model_name(i) for i in df.index], rotation=45, ha='right', fontsize=12)
ax[0].set_xlabel('')

ax[0].legend(['Concept', 'Matrix', 'Duplicate'], frameon=False, fontsize=12)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)


df = discrimination.df.groupby('model')[response_cols].mean().sort_values('concept_response', ascending=True)
df.plot(kind='bar', stacked=False, color=['#B53A3D', '#508FA3', '#F3E4D0'], edgecolor='black', ax=ax[1])
ax[1].set_title('Discrimination', fontsize=16)
ax[1].set_ylabel('% of responses', fontsize=14)
ax[1].set_yticks(np.arange(0, 0.9, 0.2), ['{:.0f}%'.format(i*100) for i in np.arange(0, 0.9, 0.2)], fontsize=12)
ax[1].set_xticks(range(df.shape[0]), [get_model_name(i) for i in df.index], rotation=45, ha='right', fontsize=12)
ax[1].set_xlabel('')
ax[1].legend(['Concept', 'Matrix', 'Duplicate'], frameon=False, fontsize=12)
ax[1].legend([], frameon=False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

plt.show()

#df = discrimination.df[discrimination.df['model'] == 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo']
plt.figure(figsize=(10, 5))
item_df = discrimination.df.groupby('item_main_id')[['concept_response', 'matrix_response', 'duplicate_response', 'other_response']].mean()
item_df.sort_values('concept_response', ascending=False).plot(kind='bar', stacked=True)
plt.xticks([])
plt.show()

item_df.hist(bins=20, figsize=(10, 5))
# %%
def plot_most_common(item_df, choice, top=3):
    df = item_df.sort_values(by=f'{choice}_response', ascending=False)[:top]
    for i in range(top):    
        print(f'Item: {df.index[i]} {choice}')
        discrimination.dataset.plot(df.index[i], f'Item: {df.index[i]} {choice}: {df.iloc[i][f'{choice}_response'] * 100:.2f}%')

plot_most_common(item_df, 'duplicate', top=3)
# %%
import seaborn as sns
df_item_concept = pd.DataFrame({
    'Discrimination': discrimination.df.groupby('item_main_id')['concept_response'].mean(),
    'Generation': generation.df.groupby('item_main_id')['concept_response'].mean(),
    'Recognition': recognition.df.groupby('item_main_id')['concept_response'].mean(),
})
sns.heatmap(df_item_concept[1:].corr(), cmap='coolwarm', vmin=-1, vmax=1, annot=True, square=True, mask=np.tril(df_item_concept[1:].corr()) == 0)
plt.tight_layout()
plt.show()
# %%
fig, ax = plt.subplots(3, 3, figsize=(15, 10))
for i, model in enumerate(df_con['model'].unique()):
    df = df_con[df_con['model'] == model]
    sns.heatmap(df[['Discrimination', 'Recognition', 'Generation']].corr(), cmap='coolwarm', vmin=-1, vmax=1, annot=True, ax=ax[i//3, i%3])
    ax[i//3, i%3].set_title(names.get(model.split('/')[1], model))# %%

pd.DataFrame({
    'Discrimination': discrimination.df.groupby('item_id')['concept_response'].mean(),
    'Generation': generation.df.groupby('item_id')['concept_response'].mean(),
    'Recognition': recognition.df.groupby('item_id')['concept_response'].mean(),
})

#%% 
mc_lp = discrimination.df.groupby('choice')['logprobs'].mean()
mc_lp_err = discrimination.df.groupby('choice')['logprobs'].sem()

abcd_lp = recognition.df.groupby('concept_response')['logprobs'].mean()[1]
abcd_lp_err = recognition.df.groupby('concept_response')['logprobs'].sem()[1]

oe_concept_lp = generation.df.groupby('concept_response')['logprobs'].mean().loc[1]
oe_concept_lp_err = generation.df.groupby('concept_response')['logprobs'].sem().loc[1]
or_matrix_lp = generation.df.groupby('matrix_response')['logprobs'].mean().loc[1]
or_matrix_lp_err = generation.df.groupby('matrix_response')['logprobs'].sem().loc[1]
or_duplication_lp = generation.df.groupby('duplicate_response')['logprobs'].mean().loc[1]
or_duplication_lp_err = generation.df.groupby('duplicate_response')['logprobs'].sem().loc[1]
oe_other_lp = generation.df.groupby('other_response')['logprobs'].mean().loc[1]
oe_other_lp_err = generation.df.groupby('other_response')['logprobs'].sem().loc[1]
# %%
