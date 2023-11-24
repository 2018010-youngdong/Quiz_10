import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
filename = './data/09_irisdata.csv'
column_names = ['sepal_length','sepal-width','petal-length','petal-width','class']
data=pd.read_csv(filename, names=column_names)
print(data.groupby('class').size())
print(data.shape)
print(data.describe())
scatter_matrix(data)
plt.savefig("./data/scatter_plot.png")
X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)
model = DecisionTreeClassifier()
kfold = KFold(n_splits=10, random_state=5, shuffle=True)
results = cross_val_score(model, X, Y, scoring='accuracy', cv=kfold)
print(results.mean())
