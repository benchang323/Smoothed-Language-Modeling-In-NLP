# Smoothed Language Modeling In NLP

## Overview

Smoothed Language Modeling In NLP is a Natural Language Processing (NLP) project that focuses on the development of a language model. This project leverages trigram models, log-linear modeling techniques, and incorporates sophisticated smoothing methods for enhanced language prediction. The project delves deep into tokenization, sentence and word state handling, as well as integrating external features like lexicons. Additionally, it encompasses model evaluation methods, hyperparameter optimization, and even includes a spam detection component.

## Features

- **Trigram Language Modeling**: Implements advanced trigram models for predicting word sequences, demonstrating the model's predictive power.
- **Smoothing Techniques**: Incorporates various smoothing methods, including backoff smoothing, direct estimation, and advanced techniques like Good-Turing smoothing, to refine model accuracy.
- **Log-linear Modeling**: Utilizes the PyTorch framework for constructing complex log-linear models, integrating external features like lexicons.
- **Comprehensive Model Evaluation**: Employs multiple methods for model assessment, such as perplexity calculations, sampling, and evaluation against real-world spam detection.
- **Hyperparameter Optimization**: Includes a mechanism for tuning model parameters, enhancing performance metrics and output accuracy.
- **Spam Detection**: Features a spam detection component that can identify and filter out spam text effectively.

## Files

- `model.py`: Core implementation of the trigram language model, log-linear modeling, and spam detection.
- `data_loader.py`: Handles loading and preprocessing of datasets.
- `evaluation.py`: Functions dedicated to model evaluation, performance measurement, and spam detection evaluation.
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
