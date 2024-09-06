# Brain Connectivity Age-Group Classification

# Brain Connectivity Age Group Classification

This project uses brain imaging data to classify subjects into different age groups based on functional connectivity patterns. The analysis utilizes the Nilearn library and explores various connectivity measures such as correlation, partial correlation, and tangent space embedding.

## Project Overview

The original task involves using fMRI data from the Nilearn library to predict age groups (children vs adults) based on functional brain connectivity. We use the following steps:

1. **Data Loading**: Fetching brain development fMRI dataset.
2. **Signal Extraction**: Extracting signals from brain regions using the MSDL atlas.
3. **Connectivity Analysis**: Calculating different types of connectivity matrices (correlation, partial correlation, tangent) for classification.
4. **Classification**: Using an SVM classifier with cross-validation to predict age groups.
5. **Performance Comparison**: Comparing different types of connectivity measures for their predictive power.

### Extension

In addition to running the original analysis, I extended the notebook by:
- Comparing multiple types of functional connectivity (correlation, partial correlation, tangent) to assess their predictive accuracy.
- Adding a detailed pipeline for cross-validation and grid search to tune the model's hyperparameters.
- Visualizing the classification performance for each type of connectivity matrix to identify the best-performing method.

## Installation

To run this notebook, you'll need to install the following dependencies:
- `nilearn`
- `scikit-learn`
- `matplotlib`

## Usage

Run the Jupyter notebook to see the analysis and results. You can find the code in the notebook, which compares different connectivity measures for predicting age groups.
