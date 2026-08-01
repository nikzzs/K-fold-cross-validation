from preparation import dataset
import numpy as np
from sklearn.model_selection import cross_val_score
from pipeline import pipeline

X_train, X_test, y_train, y_test = dataset()
pipe_lr = pipeline()

scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=-1)
print(f'CV accuracy scores: {scores}')

print(f'CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')