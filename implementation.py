from preparation import dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pipeline import pipeline

X_train, X_test, y_train, y_test = dataset()
pipe_lr = pipeline()

kfold = StratifiedKFold(n_splits=10).split(X_train, y_train)
scores = []

for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print(f'Fold: {k+1:02d}')
    print(f'Class distribution: {np.bincount(y_train[train])}')
    print(f'Accuracy: {score:.3f}')

mean_accuracy = np.mean(scores)
standard_accuracy = np.std(scores)

print(f'\nCV accuracy: {mean_accuracy:.3f} +/- {standard_accuracy:.3f}')