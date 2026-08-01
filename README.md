# K-fold-cross-validation
Implementation of k-fold cross-validation, which can help us to obtain reliable estimates of the model’s generalisation performance, that is, how well the model performs on unseen data.

Used the Breast Cancer Wisconsin dataset, which contains 569 examples of malignant and benign tumor cells.
The first two columns in the dataset store the unique ID numbers of the examples and the corresponding diagnoses (M = malignant, B = benign), respectively.
Columns 3-32 contain 30 real-valued features that have been computed from digitized images of the cell nuclei, which can be used to build a model to predict whether a tumor is benign or malignant.
The Breast Cancer Wisconsin dataset has been deposited in the UCI Machine Learning Repository, and more detailed information about this dataset can be found at https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic).

Since the features in the Breast Cancer Wisconsin dataset are measured on various different scales, we will standardize the columns in the Breast Cancer Wisconsin dataset before we feed them to a linear classifier, such as logistic regression.
Compressed the data from the initial 30 dimensions into a lower two-dimensional subspace via principal component analysis (PCA), a feature extraction technique for dimensionality reduction.
