# 🌿 Plant Species Identifier

A deep learning prototype that looks at a photo of a leaf and predicts the plant's **species**, then shows its **genus**, **family**, **characteristics**, **uses** and **habitat**.

Built as a university prototype (BS Computer Science, University of Agriculture Faisalabad) to explore how image classification can help identify crops.

![App demo](images/demo.png)

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [How It Works](#how-it-works)
4. [Dataset](#dataset)
5. [Model](#model)
6. [Results](#results)
7. [Characteristics Table](#characteristics-table)
8. [Project Structure](#project-structure)
9. [Installation and Usage](#installation-and-usage)
10. [How to Retrain the Model](#how-to-retrain-the-model)
11. [Limitations](#limitations)
12. [Future Work](#future-work)
13. [Tech Stack](#tech-stack)
14. [Glossary](#glossary)
15. [Credits](#credits)

---

## Overview

Identifying a plant by eye takes expertise. This project shows that a small model, trained in about ten minutes on a free GPU, can recognise **14 crop species** from a leaf photo with about **95% accuracy** on photos it had never seen.

- **Task type:** supervised, multi-class **image classification** (the answer is one category out of 14, not a number)
- **Approach:** transfer learning with a pretrained MobileNetV2 model
- **Interface:** a web app built with Streamlit

The 14 species are: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry and Tomato.

## Features

- Upload a JPG or PNG photo of a leaf
- See the **top 3 predicted species** with confidence percentages
- See the **genus, family, characteristics, uses and habitat** of the best match
- A **"not sure" message** appears when the confidence is below 70%, instead of a risky guess
- Runs locally on a normal laptop

## How It Works

```
Upload photo  ->  Resize to 224x224  ->  MobileNetV2 model  ->  Top 3 species + confidence  ->  Look up genus and characteristics
```

1. The user uploads a photo in the web app.
2. The photo is converted to RGB and resized to 224 x 224 pixels, the size the model was trained on.
3. The model outputs 14 probabilities, one per species. The app keeps the three highest.
4. If the best probability is below 70%, the app says it is not sure. Otherwise it looks up the winning species in `data/plants.csv` and displays its details.

The model predicts **only the species name**. Genus and characteristics come from the lookup table, not from the model.

## Dataset

| Item | Detail |
|---|---|
| Source | PlantVillage dataset on Kaggle (uploaded by Abdallah Ali), colour images only |
| Original work | PlantVillage project (Hughes and Salathe, 2015) |
| Species used | 14 crops |
| Images per species | 350 (a balanced set) |
| Total images | 4,900 |
| Split | 80% train (3,920), 10% validation (490), 10% test (490) |

**Preparation steps** (see `scripts/prepare_data.py`):

1. The original dataset has one folder per crop *and disease*, for example `Apple___Apple_scab`. All disease folders of one crop were merged into a single species folder, so the model learns the **species**, not the disease.
2. Each species was limited to 350 randomly chosen images so no species is favoured.
3. Images were split into train, validation and test folders. The test images were never used during training.

The full dataset is not stored in this repository because of its size. Download it from Kaggle (search "PlantVillage dataset") if you want to retrain.

## Model

| Setting | Value |
|---|---|
| Base model | MobileNetV2, pretrained on ImageNet |
| Method | Transfer learning: base layers frozen, new classifier layer trained |
| Input size | 224 x 224 x 3 |
| Data augmentation | Random horizontal flip, rotation (10%), zoom (10%) |
| Classifier head | Global average pooling, dropout 0.2, dense layer with 14 outputs (softmax) |
| Optimiser | Adam, learning rate 0.001 |
| Loss | Sparse categorical cross-entropy |
| Epochs / batch size | 10 / 32 |
| Training hardware | Free Google Colab T4 GPU |

The training notebook is in `notebooks/train_model.ipynb` and includes the printed training output.

## Results

| Metric | Value |
|---|---|
| Validation accuracy (final epoch) | about 97.4% |
| **Test accuracy** | **about 95%** (467 of 490 test photos correct) |

The test accuracy is the honest number, because those photos were never seen during training or tuning.

![Confusion matrix](images/confusion_matrix.png)

**What the confusion matrix shows**

- Perfect on the test set: Orange, Raspberry and Squash (35 out of 35).
- Weakest species: Strawberry, about 86% correct (30 out of 35).
- Most mistakes are single photos scattered across species. No single pair of species is repeatedly confused.

**Informal test outside the 14 species**

Three photos of plants that are not in the 14 species were uploaded (a leafy plant with small flowers, plain green leaves, a sunflower).

- 2 of 3 were correctly rejected with the "not sure" message.
- 1 received a confident wrong answer (Strawberry, about 90%).

This shows the confidence threshold helps but cannot fully protect against unfamiliar plants. See [Limitations](#limitations).

## Characteristics Table

`data/plants.csv` holds the details shown for each predicted species.

| Column | Meaning |
|---|---|
| `class_name` | The name the model predicts (must match `model/class_names.txt`) |
| `species_name` | Scientific (Latin) name |
| `genus` | Genus, the first part of the scientific name |
| `family` | Plant family |
| `characteristics` | Short description of how the plant looks |
| `uses` | Common uses |
| `habitat` | Where and how it grows |

Simplification: Cherry is listed as *Prunus avium* (sweet cherry), while the dataset labels cherries as "including sour".

## Project Structure

```
plant-species-identifier/
├── README.md                 Project documentation (this file)
├── app.py                    Streamlit web app
├── requirements.txt          Python libraries needed to run the app
├── .gitignore                Files Git should ignore
├── model/
│   ├── plant_model.keras     Trained model
│   └── class_names.txt       The 14 species names in the model's order
├── data/
│   └── plants.csv            Genus and characteristics table
├── notebooks/
│   └── train_model.ipynb     Google Colab training notebook with results
├── scripts/
│   └── prepare_data.py       Merges folders, balances classes, splits the data
└── images/
    ├── demo.png              App screenshot
    └── confusion_matrix.png  Test-set confusion matrix
```

## Installation and Usage

**Requirements:** Python 3.10 or 3.11 is recommended.

```bash
# 1. Clone the repository
git clone https://github.com/Aqdas693/plant-species-identifier.git
cd plant-species-identifier

# 2. (Optional) create a separate environment
conda create -n plantapp python=3.11 -y
conda activate plantapp

# 3. Install the libraries
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. Upload a clear photo of a single leaf.

**Tips**

- If the model fails to load, your TensorFlow version may be older than the one used for training. Install a recent version.
- The "not sure" cut-off is the `CONFIDENCE_THRESHOLD` value near the top of `app.py`.

## How to Retrain the Model

1. Download the PlantVillage colour images from Kaggle.
2. Edit the two paths at the top of `scripts/prepare_data.py` and run `python scripts/prepare_data.py`. This creates a balanced `train/val/test` dataset.
3. Zip the output folder and upload it to Google Drive (or directly to Colab).
4. Open `notebooks/train_model.ipynb` in Google Colab with a GPU runtime and run the cells in order.
5. Download the new `plant_model.keras` and `class_names.txt` into the `model/` folder.

## Limitations

- **Clean training photos.** PlantVillage images are single leaves on plain backgrounds. Real field photos with soil, hands or other plants are harder.
- **No "Other" option.** The model must always choose one of 14 species, so an unknown plant can still get a confident wrong answer.
- **Only 14 species.** It cannot recognise anything outside the list.
- **Fixed characteristics.** The details come from a hand-checked table, not from the model.
- **Small prototype.** Results were measured on a small test set (490 photos).

## Future Work

- Add an **"Other / not in list"** class trained on non-target plants, flowers and backgrounds
- Add **real-world field photos** to the training data
- Support more species and **fine-tune** the pretrained layers
- Build a **mobile app** and offline mode
- Add **expert verification** of results

## Tech Stack

| Tool | What it is used for |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Building, training and running the model |
| MobileNetV2 | Pretrained image model used as the starting point |
| Streamlit | Turning the Python script into a web app |
| pandas | Reading the characteristics table |
| NumPy | Handling images as numbers |
| Pillow (PIL) | Opening and resizing uploaded images |
| scikit-learn, matplotlib | Confusion matrix and charts (in the notebook) |
| Google Colab | Free GPU for training |
| Kaggle | Source of the dataset |

## Glossary

| Term | Easy meaning |
|---|---|
| Classification | The model picks one category from a fixed list |
| Supervised learning | The model learns from examples that come with the correct answer |
| Transfer learning | Reusing a model already trained on millions of images and retraining only its last layer |
| Pretrained model | A model someone else already trained (here, on ImageNet) |
| Epoch | One full pass through all training images |
| Validation set | Images used to check progress during training |
| Test set | Images kept aside for the final, honest score |
| Overfitting | Memorising the training photos but failing on new ones |
| Confusion matrix | A table showing which species get mixed up with which |
| Confidence | How sure the model says it is, as a percentage |
| Data augmentation | Randomly flipping, rotating and zooming images to give the model more variety |

## Credits

- **Author:** [Aqdas693](https://github.com/Aqdas693), BS Computer Science, University of Agriculture Faisalabad
- **Dataset:** PlantVillage (Hughes and Salathe, 2015), as shared on Kaggle by Abdallah Ali
- **Pretrained model:** MobileNetV2, trained on ImageNet
- Built as an educational prototype
