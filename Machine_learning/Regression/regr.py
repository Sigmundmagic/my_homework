import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
%matplotlib inline
from google.colab import files

# noisysine dataset

files.upload()
df = pd.read_csv("noisysine.csv")

X = df.drop("y", axis=1).values
y = df.y.values


def plot_regression(X, y, deg = 1, alpha = 0.01, method = "simple"):
  poly = PolynomialFeatures(degree = deg)
  X_polynomial = poly.fit_transform(X)
  if method == "Ridge":
    model = Ridge(alpha=alpha)
  elif method == "Lasso":
    model = Lasso(alpha= alpha)
  else:
    model = LinearRegression()
  model.fit(X_polynomial,y)

  #you need to fill X space to draw a pretty plot
  X_plot = np.linspace(X.min() - 10, X.max() + 10, 500).reshape(-1,1)
  X_plot_polynomial = poly.fit_transform(X_plot)
  y_plot = model.predict(X_plot_polynomial)
  plt.plot(X_plot, y_plot, label = "Degree = " + str(deg) + 
            " R2 = " + str(round(model.score(X_polynomial,y),2)))    
  plt.legend(loc='lower left')
  print("for degree = " + str(i) + " R2 = " + str(round(model.score(X_polynomial,y), 4)))
  print("number of features used in the model is " + str(sum(model.coef_ != 0)) + " out of " + str(len(model.coef_)))
  
  
  
# Linear and polynimial

plt.figure(figsize=(10,7))
plt.xlim([X.min() - 10, X.max() + 10])
plt.ylim([y.min() - 10, y.max() + 10])
plt.scatter(X, y, color = "black", label = "Training points")
for i in [1,2,3,5]:
  plot_regression(X, y, deg = i)
plt.show()

# Ridge regression

plt.figure(figsize=(10,7))
plt.xlim([X.min() - 10, X.max() + 10])
plt.ylim([y.min() - 10, y.max() + 10])
plt.scatter(X, y, color = "black", label = "Training points")
for i in [1,2,3,5]:
  plot_regression(X, y, deg = i, alpha = 0.0001,method = "Ridge")
plt.show()

# Lasso regression

plt.figure(figsize=(10,7))
plt.xlim([X.min() - 10, X.max() + 10])
plt.ylim([y.min() - 10, y.max() + 10])
plt.scatter(X, y, color = "black", label = "Training points")
for i in [1,2,3,5]:
  plot_regression(X, y, deg = i, alpha = 0.0001,method = "Lasso")
plt.show()

# hydrodynamics dataset

files.upload()
df = pd.read_csv("hydrodynamics.csv")

X = df.drop("y", axis=1).values
y = df.y.values

def find_regression(X, y, deg = 1, alpha = 0.01, method = "simple"):
  poly = PolynomialFeatures(degree = deg)
  X_polynomial = poly.fit_transform(X)
  if method == "Ridge":
    model = Ridge(alpha=alpha)
  elif method == "Lasso":
    model = Lasso(alpha= alpha)
  else:
    model = LinearRegression()
  model.fit(X_polynomial,y)
  print("for degree = " + str(i) + " R2 = " + str(round(model.score(X_polynomial,y), 4)))
  print("number of features used in the model is " + str(sum(model.coef_ != 0)) + " out of " + str(len(model.coef_)))

# Linear and polynomial

for i in [1,2,3,5]:
  find_regression(X, y, deg = i)
  
# Ridge regression

for i in [1,2,3,5]:
  find_regression(X, y, deg = i, alpha =0.1, method = "Ridge")
  
# Lasso regretion

for i in [1,2,3,5]:
  find_regression(X, y, deg = i, alpha = 0.1, method = "Lasso")
