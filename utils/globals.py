import yaml
import os

# Define the path to the globals file
base_dir = os.path.dirname(os.path.abspath(__file__))
globals_path = os.path.join(base_dir, '..', 'globals.yml')

# Load data from YAML
with open(globals_path, "r") as stream:
    data = yaml.safe_load(stream)

# Define global variables
FILES = [data[key] for key in ["ITEMS_FILE", "RESULTS_DIR"]]
ITEMS_FILE, RESULTS_DIR = FILES