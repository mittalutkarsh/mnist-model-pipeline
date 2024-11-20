# MNIST Model Pipeline

A comprehensive CI/CD pipeline for training and validating MNIST digit classification models using multiple CNN architectures. This project demonstrates automated testing, model validation, and performance benchmarking using GitHub Actions.

## Project Overview

This project implements multiple CNN architectures for MNIST digit classification with a focus on:
- Automated testing and validation
- Parameter efficiency = 10K
- Accuracy - 95.3%
- CI/CD integration

## Model Architectures

### 1. SimpleCNN
- Basic CNN architecture
- Minimal parameters
- 2 convolutional layers
- Single fully connected layer

### 2. DeeperCNN
- Enhanced depth
- Multiple convolutional layers
- Increased channel dimensions
- Optimized for feature extraction

### 3. Model1CNN
- Dropout for regularization
- Intermediate dense layers
- Reduced padding in convolutions
- Optimized for preventing overfitting

### 4. Model2CNN (Best Performing)
- Residual connections
- Batch normalization
- Advanced architecture
- Best accuracy-to-parameter ratio

## Requirements

- Python 3.9+
- PyTorch 2.2.0
- torchvision 0.17.0
- pytest 7.4.0
- numpy 1.24.3
- matplotlib 3.7.1
- tqdm 4.65.0

## Installation