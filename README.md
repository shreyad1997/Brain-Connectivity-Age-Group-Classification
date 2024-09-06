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

## References

1. Nilearn. (n.d.). Functional connectivity predicts age group. Retrieved from [Nilearn Documentation](https://nilearn.github.io/stable/index.html)
2. Dadi, K., Rahim, M., Abraham, A., Chyzhyk, D., Milham, M. P., Thirion, B., & Varoquaux, G. (2019). Benchmarking functional connectome-based predictive models for resting-state fMRI. *NeuroImage*, 192, 115-134. doi: [10.1016/j.neuroimage.2019.02.062](https://doi.org/10.1016/j.neuroimage.2019.02.062)
3. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
