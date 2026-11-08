# Controlled Gradient-Flow Ablation in Deep Neural Networks

A controlled PyTorch study investigating how **activation functions, network depth, and weight initialization** affect gradient magnitudes across layers in deep neural networks.

## Overview

The study uses three ablation experiments on multilayer perceptrons (MLPs):

| Study | Variable              | Configurations      |
| ----- | --------------------- | ------------------- |
| A     | Activation function   | Sigmoid, Tanh, ReLU |
| B     | Network depth         | 4, 8, 16, 32 layers |
| C     | Weight initialization | Xavier, He          |

## Experimental Setup

- **Framework:** PyTorch
- **Input dimension:** 20
- **Hidden width:** 64
- **Batch size:** 128
- **Random seeds:** 11, 23, 37
- **Loss function:** `BCEWithLogitsLoss`
- **Measurements:** Weight-gradient L2 norm and RMS

Each experiment performs a single forward and backward pass using synthetic inputs and binary targets. No optimizer step is performed.

## Key Findings

The experiments are designed to examine how activation saturation, increasing network depth, and initialization choices influence gradient magnitudes across layers.

Results should be interpreted as **single-pass gradient diagnostics**, not evidence of training accuracy, convergence, or generalization. Numerical conclusions should be reported only after verification against the final experiment outputs.

## Repository Contents

```txt
├── paper                       # research manuscript materials
│   └── controlled_gradient_flow.pdf
├── results                     # generated summaries and plots
│   ├── figures
│   │   ├── activation_ablation.png
│   │   └── depth_scaling.png
│   ├── gradient_measurements.csv
│   └── run_summary.csv
├── src
│   ├── __init__.py
│   └── builds.py
├── .gitignore
├── README.md
├── run_experiments.ipynb       # experiment implementation and analysis
└── test_experiments.py			# test experiments by running: "pytest -q" in terminal
```

## Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open `run_experiments.ipynb` and run all cells from a fresh kernel.

## Limitations

- Three random seeds
- Synthetic data
- One backward pass per run
- No evaluation of model accuracy or training convergence

## Author

**Rajesh Phulwaria** — Independent Researcher

- [GitHub](https://github.com/RajeshPhulwaria006)
- [LinkedIn](https://www.linkedin.com/in/rajesh-phulwaria-b61093315/)
- [Publication](https://doi.org/10.5281/zenodo.24053276)
