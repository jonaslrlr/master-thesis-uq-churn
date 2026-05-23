# Uncertainty Quantification for Deep Learning Churn Prediction

Code for the master's thesis *Evaluating the Effect of Uncertainty Quantification in Deep Learning for Churn Prediction*.

Jonas Lörler · KU Leuven / Ghent University · 2025–2026
Supervisors: Prof. Seppe vanden Broucke, Yameng Guo

## Overview

Four uncertainty quantification methods applied to TabNet for customer churn prediction across six datasets, evaluated for whether the uncertainty signal adds predictive value beyond the base probability in a logistic regression reranker.

- **Methods**: Monte Carlo Dropout, last-layer Laplace approximation, Evidential Deep Learning, Conformal Prediction
- **Datasets**: Bank, Cell2Cell, Telco, Delft, CDR, Chile
- **Backbone**: TabNet (Dreamquark-AI implementation, forked to support dropout in the GLU and attention paths)

## Repository structure

    src/
    ├── pytorch_tabnet/        # Forked TabNet with GLU + attention dropout
    └── thesis_uq/
        ├── data/              # Dataset loaders
        ├── models/            # Baseline, MC Dropout, Laplace, EDL, Conformal
        ├── gridsearch/        # Hyperparameter selection
        ├── eval/              # Evaluation on held-out test set
        ├── metrics/           # Ranking + uncertainty quality metrics
        └── plots/             # UQ scatter plots
    reports/
    ├── best/                  # Best hyperparameters per method × dataset
    ├── eval/                  # Per-seed test results
    ├── results/               # Per-config gridsearch results
    ├── uq_scores/             # NPZ files with probabilities + uncertainties
    └── plots/                 # Scatter plots from eval runs
