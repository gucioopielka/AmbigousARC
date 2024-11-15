import yaml
import os

def get_model_name(model_id):
    return MODEL_DICT.get(model_id, model_id)

# Define the path to the globals file
base_dir = os.path.dirname(os.path.abspath(__file__))
globals_path = os.path.join(base_dir, '..', 'globals.yml')

# Load data from YAML
with open(globals_path, "r") as stream:
    data = yaml.safe_load(stream)

# Define global variables
FILES = [os.path.join(base_dir, '..', data[file]) for file in data.keys() if 'FILE' in file or 'DIR' in file]
ITEMS_FILE, RESULTS_DIR = FILES

# Define the names of the models
MODEL_IDS = data['MODEL_IDS']
MODEL_NAMES = data['MODEL_NAMES']
MODEL_DICT = {id: name for id, name in zip(MODEL_IDS, MODEL_NAMES)}