# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commonly Used Commands

### Installation
```bash
pip install -r requirements.txt
# Optional for GNN baselines (if baselines directory is present)
pip install torch_geometric
```

### Loading and Visualizing Data
See `resplan_utils.py` for loading and plotting functions. Example usage:
```python
import pickle
from resplan_utils import plot_plan, plot_plan_and_graph

with open("ResPlan.pkl", "rb") as f:
    data = pickle.load(f)

plan = data[0]
plot_plan(plan)
plot_plan_and_graph(plan)
```

### Running the Demo Notebook
```bash
jupyter notebook ResPlan_demo.ipynb
# or
jupyter lab ResPlan_demo.ipynb
```

### Training, Inference, and Evaluation
Scripts are located in the `pipeline/` directory:
- Training: `python pipeline/train.py`
- Inference: `python pipeline/infer.py`
- Evaluation: `python pipeline/evaluate.py`
Refer to the scripts' command-line arguments for configuration options.

### Using Data Splits
The canonical train/val/test splits are in `split.json`. Augmented plans are also provided.
```python
import json
with open("split.json") as f:
    splits = json.load(f)
train_ids = set(splits["train"])
val_ids   = set(splits["val"])
test_ids  = set(splits["test"])
aug_ids   = set(splits["augmented"])
```

## Project Structure

- **Root Directory**:
  - `ResPlan.pkl`: Main dataset file (17,000 floor plans as pickle).
  - `split.json`: Canonical train/val/test splits and augmented IDs.
  - `croissant.json`: Metadata in JSON-LD format (for data provenance).
  - `resplan_utils.py`: Core utilities for loading, plotting, and manipulating plans.
  - `ResPlan_demo.ipynb`: Interactive demo notebook demonstrating usage.
  - `extract_to_json.py`, `visualize_floorplan.py`: Helper scripts for conversion and visualization.
  - `requirements.txt`: Python dependencies.

- **`pipeline/`**: Contains training, inference, and evaluation pipelines.
  - `config.py`: Configuration settings.
  - `dataset.py`: Dataset loading and preprocessing for PyTorch.
  - `model.py`: Model architectures (GNNs, MLPs, etc.).
  - `train.py`: Training loop.
  - `infer.py`: Inference script.
  - `evaluate.py`: Evaluation metrics and reporting.

- **`checkpoints/`**: Saved model checkpoints and inference outputs (generated during training/evaluation).

- **`__pycache__/`, `.venv/`**: Python cache and virtual environment (local development).

## Notes
- The dataset is in pickle format containing Shapely geometries and NetworkX graphs.
- Room and graph attributes are detailed in the README (semantic labels, edge types, etc.).
- For baseline experiments, refer to the `baselines/` directory if present (not included in this repository clone).
- Always respect the data license (CC BY 4.0) and code license (MIT) when using or redistributing.