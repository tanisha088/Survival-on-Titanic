# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train_titanic.csv')
dataset1 = pd.read_csv('test_titanic.csv')


char_cols = dataset.dtypes.pipe(lambda x: x[x == 'object']).index
for c in char_cols:
    dataset[c] = pd.factorize(dataset[c])[0]
dataset.info()
char_cols = dataset1.dtypes.pipe(lambda x: x[x == 'object']).index
for c in char_cols:
    dataset1[c] = pd.factorize(dataset1[c])[0]
dataset1.info()

#print(dataset.describe())
#X_train.info()
#print(dataset.head(20))
dataset =dataset.fillna(dataset.mean())
dataset1 = dataset1.fillna(dataset.mean())
print(dataset.isnull().sum())
print(dataset1.isnull().sum())

"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(dataset.values[:, 0:12])
dataset.values[:,0:12]=imputer.transform(dataset.values[:,0:12])
null_data = dataset[dataset.isnull().any(axis=1)]
dataset.isnull().sum().sum()


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(dataset1.values[:, 0:12])
dataset1.values[:,0:12]=imputer.transform(dataset1.values[:,0:12])
dataset1.isnull().sum().sum()
"""
null_data = dataset1[dataset1.isnull().any(axis=1)]


y_train = dataset.iloc[:, 1]
del dataset['Survived']
X_train = dataset
dataset.info()
dataset1.info()
X_test = dataset1


"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
"""
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
"""
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

new1  = pd.DataFrame(y_pred)