# Credit scoring - Interpretability, Stability and Fairness course final project

This GitHub repository contains the code for the final project of the Introduction to Time Series course from the MScT Data Science and AI for Business at X-HEC.

The objective of this project is to apply time series analysis methods to forecast the quality of the air in Paris from a Kaggle dataset.

## Repository structure
```
project/
├── data/                 # Raw and processed datasets
├── notebooks/            # Jupyter notebooks for steps
├── src/                  # Source code for model training and useful functions
├── models/               # Saved trained models
├── results/              # Results, figures, and outputs
├── requirements.txt      # Python dependencies
└── pyproject.toml
```



## Installation & Usage

### Development
To reproduce the development environment, follow these steps:

0. **(Prerequisite)** Have ```uv``` installed. See [the project's website](https://docs.astral.sh/uv/) for more information. In your terminal (MacOS and Linux users), run 
```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Clone the project:
```zsh
git clone https://github.com/auggy-ntn/paris-air-quality-prediction
```

2. In the project's workspace run the following command to synchronize your environment with the project's development requirements:
```zsh
uv sync --dev
```
You are all set!

Alternatively, if you don't want to use ```uv```, you can run the following command:
```zsh
pip install -r requirements.txt
```

### Developing with uv

If you work on the project and want to add a package, simply run
```zsh
uv add <package>
``` 
which will update the ```pyproject.toml``` file and the ```uv.lock``` file used by ```uv``` to sync the environment when you run ```uv sync```.

To generate the updated ```requirements.txt``` file, run the following command
```zsh
uv pip freeze > requirements.txt
```

Commit and push these new files to GitHub for others to replicate your environment.