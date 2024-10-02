from pathlib import Path
import yaml

with open("globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(ITEMS_FILE,) = (
    Path(z)
    for z in [
        data["ITEMS_FILE"],
    ]
)
