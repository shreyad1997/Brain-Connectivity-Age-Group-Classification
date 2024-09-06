
"""# Programming Task Part 2 - Functional connectivity predicts age group

The main aim of the notebook is to demonstrate how brain connectivity patterns differ between children and adults. By analyzing brain scan data, it uses various machine learning techniques to classify individuals into age groups based on these patterns. The goal is to understand how brain connections change with age and to identify the most effective methods for predicting age groups using brain imaging data.

Summary (what the code does):

The code below does the following:


*   Data Loading - Fetches the brain imaging dataset (developmental fMRI).
*   Signal Extraction: Extracts time-series signals from brain regions using an atlas.
*   Functional Connectivity: Computes correlation matrices (connectomes) for each subject.
*   Label Processing: Groups subjects into different age groups based on metadata.

*  Cross-Validation: Sets up a stratified cross-validation to evaluate model performance
*   Model Training: Uses a Support Vector Machine (SVM) classifier to predict age groups based on connectome data.

*   Performance Evaluation: Evaluates the classifier's performance using cross-validation scores.
"""

"""
Functional connectivity predicts age group
==========================================

This example compares different kinds of :term:`functional connectivity`
between regions of interest : correlation, partial correlation,
and tangent space embedding.

The resulting connectivity coefficients can be used to
discriminate children from adults. In general, the tangent space embedding
**outperforms** the standard correlations:
see :footcite:t:`Dadi2019` for a careful study.

.. include:: ../../../examples/masker_note.rst

"""
!pip install nilearn

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

# %%
# Load brain development :term:`fMRI` dataset and MSDL atlas
# ----------------------------------------------------------
# We study only 60 subjects from the dataset, to save computation time.
from nilearn import datasets

development_dataset = datasets.fetch_development_fmri(n_subjects=60)

# %%
# We use probabilistic regions of interest (ROIs) from the MSDL atlas.
from nilearn.maskers import NiftiMapsMasker

msdl_data = datasets.fetch_atlas_msdl()
msdl_coords = msdl_data.region_coords

masker = NiftiMapsMasker(
    msdl_data.maps,
    resampling_target="data",
    t_r=2,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    memory="nilearn_cache",
    memory_level=1,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
).fit()

masked_data = [
    masker.transform(func, confounds)
    for (func, confounds) in zip(
        development_dataset.func, development_dataset.confounds
    )
]

# %%
# What kind of connectivity is most powerful for classification?
# --------------------------------------------------------------
# we will use connectivity matrices as features to distinguish children from
# adults. We use cross-validation and measure classification accuracy to
# compare the different kinds of connectivity matrices.

# prepare the classification pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from nilearn.connectome import ConnectivityMeasure

kinds = ["correlation", "partial correlation", "tangent"]

pipe = Pipeline(
    [
        (
            "connectivity",
            ConnectivityMeasure(
                vectorize=True,
                standardize="zscore_sample",
            ),
        ),
        (
            "classifier",
            GridSearchCV(LinearSVC(dual=True), {"C": [0.1, 1.0, 10.0]}, cv=5),
        ),
    ]
)

param_grid = [
    {"classifier": [DummyClassifier(strategy="most_frequent")]},
    {"connectivity__kind": kinds},
]

# %%
# We use random splits of the subjects into training/testing sets.
# StratifiedShuffleSplit allows preserving the proportion of children in the
# test set.
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

groups = [pheno["Child_Adult"] for pheno in development_dataset.phenotypic]
classes = LabelEncoder().fit_transform(groups)

cv = StratifiedShuffleSplit(n_splits=30, random_state=0, test_size=10)
gs = GridSearchCV(
    pipe,
    param_grid,
    scoring="accuracy",
    cv=cv,
    verbose=1,
    refit=False,
    n_jobs=2,
)
gs.fit(masked_data, classes)
mean_scores = gs.cv_results_["mean_test_score"]
scores_std = gs.cv_results_["std_test_score"]

# %%
# display the results
plt.figure(figsize=(6, 4))
positions = [0.1, 0.2, 0.3, 0.4]
plt.barh(positions, mean_scores, align="center", height=0.05, xerr=scores_std)
yticks = ["dummy"] + list(gs.cv_results_["param_connectivity__kind"].data[1:])
yticks = [t.replace(" ", "\n") for t in yticks]
plt.yticks(positions, yticks)
plt.xlabel("Classification accuracy")
plt.gca().grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()

# %%
# This is a small example to showcase nilearn features. In practice such
# comparisons need to be performed on much larger cohorts and several
# datasets.
# :footcite:t:`Dadi2019` showed
# that across many cohorts and clinical questions,
# the tangent kind should be preferred.

plt.show()

# %%
# References
# ----------
#
#  .. footbibliography::

"""# Addtional Visulations and Analysis

1. Additional Visualizations

**Connectivity Matrix Visualization**
Will help visualize the average functional connectivity matrices for different age groups which can help us understand the differences in brain connectivity patterns.

2. Comparing Algorithms

**Implementing Additional Classifiers**
Comparing the performance of multiple classifiers such as Random Forest, Gradient Boosting, and K-Nearest Neighbors with the existing SVM.

3. Quality/Robustness Assessment

**Permutation Testing** can help assess the statistical significance of the classifier's performance.
"""

# Additional Visualizations

import numpy as np
from nilearn import plotting


# Flatten the connectivity matrices
flattened_data = [matrix.flatten() for matrix in masked_data]

# Compute mean connectivity matrices for each age group
age_groups = np.unique(classes)
mean_matrices = {age: np.mean([masked_data[i] for i in range(len(classes)) if classes[i] == age], axis=0) for age in age_groups}

# Plot connectivity matrices
for age, matrix in mean_matrices.items():
    plotting.plot_matrix(matrix, title=f"Mean Connectivity Matrix for Age Group {age}", colorbar=True)
    plt.show()

# Comparing Algorithms

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Initialize classifiers
classifiers = {
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Evaluate classifiers
for name, clf in classifiers.items():
    scores = cross_val_score(clf, flattened_data, classes, cv=cv, scoring='accuracy')
    print(f"{name} Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")


# Quality/Robustness Assessment
from sklearn.model_selection import permutation_test_score

# Permutation testing for SVM
svm = LinearSVC()
score, perm_scores, p_value = permutation_test_score(svm, flattened_data, classes, cv=cv, n_permutations=100, scoring='accuracy')
print(f"SVM Permutation Test Score: {score:.2f}, p-value: {p_value:.2f}")

"""
* K-Nearest Neighbors (KNN) had the highest accuracy at 0.95, therefore it is the most effective model for predicting age groups in this dataset.

* The SVM also performed well with an accuracy of 0.93 and a statistically significant p-value of 0.01, making it a reliable model.

* The connectivity matrices highlight distinct connectivity patterns between the two age groups, aiding in understanding brain development differences.


Both KNN and SVM are strong models for this classification task, with KNN having a slight edge in performance."""

