# Smoothed Language Modeling In NLP

## Overview

The Smoothed Language Modeling Project is a cutting-edge Natural Language Processing (NLP) initiative, centered on developing a sophisticated Language Model (LM). It leverages trigram models and log-linear modeling techniques, focusing on probabilistic model applications in language. The project delves into tokenization, sentence and word state handling, and various model evaluation techniques.

## Features

- **Trigram Language Modeling**: Implements advanced trigram models for predicting word sequences, demonstrating the model's predictive power.
- **Smoothing Techniques**: Incorporates various smoothing methods, including backoff smoothing and direct estimation, to refine model accuracy.
- **Log-linear Modeling**: Utilizes the PyTorch framework for constructing complex log-linear models, integrating external features like lexicons.
- **Comprehensive Model Evaluation**: Employs multiple methods for model assessment, such as perplexity calculations and sampling, supported by a structured train/dev/test framework.
- **Hyperparameter Optimization**: Includes a mechanism for tuning model parameters, enhancing performance metrics and output accuracy.

## Files

- `model.py`: Core implementation of the trigram language model and log-linear modeling.
- `data_loader.py`: Handles loading and preprocessing of datasets.
- `evaluation.py`: Functions dedicated to model evaluation and performance measurement.
- `utils.py`: Provides utility functions for various operational needs within the project.
- `config.py`: Configuration settings and model parameters.

## Tech Stack

- **Programming Language**: Python
- **Machine Learning Framework**: PyTorch

## Libraries/Dependencies

- **PyTorch**: For building and training the log-linear models.
- **NumPy**: Used for handling numerical operations and data manipulation.
- **Matplotlib**: For visualizing model metrics and performance insights.

## Installation

```
git clone https://github.com/benchang323/Smoothed-Language-Modeling-In-NLP.git
cd Smoothed-Language-Modeling-In-NLP
```
