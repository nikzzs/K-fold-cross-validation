import pandas as pd

# Obtaining the Breast Cancer Wisconsin dataset
df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
    header=None
)

from sklearn.preprocessing import LabelEncoder
X = df.loc[:, 2:].values
y = df.loc[:, 1].values

le = LabelEncoder()

# Using a LabelEncoder object to transform the class labels from their original string representation ('M' and 'B') into integers
y = le.fit_transform(y)
print(le.classes_)

from sklearn.model_selection import train_test_split

# Dividing the dataset into a separate training dataset (80%) and a separate test dataset (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

def dataset():
    return X_train, X_test, y_train, y_test