import re
import random
import shutil
from pathlib import Path

SOURCE = r"E:\archive\plantvillage dataset\color"   # the 'color' folder
OUTPUT = r"E:\ML"        # new folder to create
MAX_PER_SPECIES = 350
random.seed(42)

def species_from_folder(name):
    crop = name.split("___")[0]           # 'Apple___Apple_scab' -> 'Apple'
    return re.split(r"[_,(]", crop)[0]    # 'Corn_(maize)' -> 'Corn'

images_by_species = {}
for folder in Path(SOURCE).iterdir():
    if folder.is_dir():
        species = species_from_folder(folder.name)
        files = [f for f in folder.iterdir()
                 if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        images_by_species.setdefault(species, []).extend(files)

for species, files in sorted(images_by_species.items()):
    random.shuffle(files)
    files = files[:MAX_PER_SPECIES]
    n = len(files)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    parts = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }
    for split, items in parts.items():
        dest = Path(OUTPUT) / split / species
        dest.mkdir(parents=True, exist_ok=True)
        for f in items:
            shutil.copy2(f, dest / f"{f.parent.name}_{f.name}")
    print(f"{species}: {n} images")