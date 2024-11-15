import os
import requests
import numpy as np
from tqdm import tqdm

X_API_KEY = 'sk-np-IChhuM0WWR6P59VLwzoj2CI4jrnrpMp4IAPnCUGzl4Y0'

def get_feature_data(model_id, sae_id, feat_idx):
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feat_idx}"
    headers = {"X-Api-Key": X_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get feature data. Status code: {response.status_code}")
    
def calculate_percentiles(frequencies):
    return (np.cumsum(frequencies) / np.sum(frequencies)) * 100

sae_dir = '/Users/gustaw/Documents/AmbigousARC/data/results/sae'
model_id = 'gemma-2-9b'
for file in os.listdir(os.path.join(sae_dir, model_id)):
    if file.endswith('npz'):
        layer = file.split('_')[-1].split('.')[0]
        width = file.split('-')[-1].split('_')[0]
        sae_id = f"{layer}-gemmascope-res-{width}"
        print(f"SAE: {sae_id}")

        n_features = np.load(os.path.join(sae_dir, model_id, file))['arr_0'].shape[-1]
        values_percentiles = np.zeros((n_features, 2, 50))
        for feat_idx in tqdm(range(n_features)):
            url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feat_idx}"
            try:
                data = get_feature_data(model_id, sae_id, feat_idx)
                act_values = data['freq_hist_data_bar_values']
                frequencies = data['freq_hist_data_bar_heights']
                percentiles = calculate_percentiles(frequencies)
                values_percentiles[feat_idx, 0] = act_values
                values_percentiles[feat_idx, 1] = percentiles
            except Exception as e:
                print(f"SAE: {sae_id}, Feature: {feat_idx}, Error: {e}")
    break
