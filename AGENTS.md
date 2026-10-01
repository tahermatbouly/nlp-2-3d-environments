# Repository Guidelines

## Project Structure & Module Organization
- Source code: `pipeline/` (training and inference scripts), `resplan_utils.py` (utility functions)
- Configuration: `requirements.txt` (dependencies), `croissant.json` (metadata)
- Data: `training-data/` (dataset), `ResPlan.pkl` (processed data), `split.json` (train/val/test splits)
- Notebooks: `ResPlan_demo.ipynb` (demonstration), `extract_to_json.py` (data extraction)
- Outputs: `checkpoints/` (model weights), `*.png` (visualizations), `__pycache__` (compiled Python)

## Build, Test, and Development Commands
- Install dependencies: `pip install -r requirements.txt`
- Train model: `python pipeline/train.py` (see pipeline/ for training scripts)
- Generate visualizations: `python visualize_floorplan.py`
- Run demo notebook: `jupyter notebook ResPlan_demo.ipynb`
- Extract data: `python extract_to_json.py`

## Coding Style & Naming Conventions
- Follow PEP 8 for Python code.
- Indentation: 4 spaces.
- Naming: `snake_case` for variables and functions, `PascalCase` for classes.
- Comments: Use docstrings for functions and classes; inline comments for complex logic.
- Formatting: Use `black` and `flake8` (if configured) for consistency.

## Testing Guidelines
- Tests are located in the `pipeline/` directory or as separate test scripts (if any).
- Run tests with: `pytest` (if configured) or execute test scripts directly.
- Test naming: `test_*.py` or functions starting with `test_`.
- Coverage: Aim to test core utility functions in `resplan_utils.py` and pipeline components.

## Commit & Pull Request Guidelines
- Commit messages: Use imperative mood, e.g., "Add feature", "Fix bug", "Update documentation".
- Reference issues: Include issue numbers if applicable (e.g., "Fixes #123").
- Pull requests: Provide a clear description of changes, link to related issues, and include screenshots for visual changes.
- Ensure code passes linting and tests before submitting.

## Additional Notes
- This repository focuses on floorplan generation using diffusion models.
- Large files (like `ResPlan.pkl`) are tracked by Git LFS; ensure LFS is installed.
- For detailed model architecture, refer to the provided PDF: `HouseDiffusion Vector Floorplan Generation via a Diffusion Modelwith Discrete and Continuous Denoising.pdf`.
