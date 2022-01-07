import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from google.colab import files

# noisysine dataset

df = pd.read_csv('noisysine.csv')
X = df.drop("y", axis=1).values
y = df["y"].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

def regression(ax, depth):
    tree = DecisionTreeRegressor(max_depth=depth)
    tree.fit(X, y)
    y_predict = tree.predict(X)
    ax.step(X, y_predict, label=f'depth={depth}')
    ax.legend() 
    
fig, ax = plt.subplots(figsize=(12,10),)
depth = 1
for i in range(2):
    for j in range(3):
        regression(ax, depth)
        depth += 1
        
ax.scatter(X, y, label='data')
ax.legend()

# spam dataset

df_spam = pd.read_csv("spam.csv")
X = df_spam.drop("label", axis=1).values
y = df_spam["label"].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

def plot_trees(depth, ax):
    tree = DecisionTreeClassifier(max_depth=depth)
    tree.fit(X_train, y_train)
    y_proba_tree = tree.predict_proba(X_test)[:,1]

    fpr_2, tpr_2, _ = roc_curve(y_test, y_proba_tree, pos_label=1)
    roc_auc = auc(fpr_2, tpr_2) 

    
    ax.plot([0, 1], [0, 1], "--")   

    ax.set_xlim(-0.05, 1)
    ax.set_ylim(0,1.05)

    ax.plot(fpr_2,tpr_2, label="Depth = " + str(tree.max_depth) + " AUC = " + str(roc_auc))

    ax.legend(loc = "lower right")

fig, ax = plt.subplots(figsize=(12,10))
depth = 1
for i in range(10):
    plot_trees(depth, ax)
    depth += 1
    
# forest =)

forest = RandomForestClassifier(n_estimators=50)
forest.fit(X_train, y_train)
y_proba_forest = forest.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_proba_forest)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,10))
plt.plot([0, 1], [0, 1], "--")    

plt.xlim(-0.05, 1)
plt.ylim(0,1.05)

plt.plot(fpr,tpr, label = " AUC = " + str(roc_auc))

plt.legend(loc = "lower right")
plt.show()

# catboost classifier

!pip install catboost

from catboost import CatBoostClassifier

cat = CatBoostClassifier(verbose=False)
cat.fit(X_train, y_train)
y_proba_cat = cat.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_proba_cat)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,10))
plt.plot([0, 1], [0, 1], "--")    

plt.xlim(-0.05, 1)
plt.ylim(0,1.05)

plt.plot(fpr,tpr, label = " AUC = " + str(roc_auc))

plt.legend(loc = "lower right")
plt.show()

