import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump, load

df = pd.read_csv("./data/data.csv")
X = df.drop(columns=['Serial No.','Chance of Admit '], axis=1)
y = df["Chance of Admit "]

X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2)

Lr = LinearRegression()
Lr.fit(X_train, Y_train)

# dump(Lr, "./model/AdmissionModel.joblib")
# print(Lr.score(x_test,y_test))