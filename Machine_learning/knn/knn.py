#import packages:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

df = pd.read_csv("cancer.csv")
df.head()

#Collect dataset (X) and labels (y):

X = df.drop("label", axis = 1).values
y = df["label"].values
#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

X_test.shape

#create new classifier
knn = KNeighborsClassifier(n_neighbors=9)

#train on train
knn.fit(X_train, y_train)

#test on test
score = knn.score(X_test, y_test)
print("The number of neighbors is ", knn.n_neighbors,"| Accuracy is ", round(score, 4))

#Drawing ROC_curve:

#.predict_proba outputs probabilities of classes
y_proba = knn.predict_proba(X_test)

#calculate false positive rates and true positive rates
fpr, tpr, _ = roc_curve(y_test, y_proba[:,1], pos_label='M')

#calculate auc
roc_auc = auc(fpr, tpr)

#plot new figure
plt.figure()

#draw diagonal line
plt.plot([0, 1], [0, 1], linewidth=2, linestyle='--')

#plot roc
plt.plot(fpr, tpr, linewidth=2, label='ROC curve (area = %0.2f)' % roc_auc)

#scale the picture
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])

#add caption
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

#output the picture
plt.show()

#Scaling the dataset:

from sklearn.preprocessing import RobustScaler
#create new scaler
scaler = RobustScaler()

#train the scaler
X_train_scaled = scaler.fit_transform(X_train)

#apply the scaler
X_test_scaled = scaler.transform(X_test)

# 1st task: 

#Calculate accuracy and draw ROC curves for two datasets (cancer.csv and spam.csv) with and without scaling for 1 to 10 neighours.

#There should be 40 plots in total.

#Cancer task

#import packages:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from google.colab import files

files.upload()
df = pd.read_csv("cancer.csv")
df.head()

X = df.drop("label", axis = 1).values
y = df["label"].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

X_test.shape

for i in range(1,11):
  knn = KNeighborsClassifier(n_neighbors=i)

  #train on train
  knn.fit(X_train, y_train)

  #test on test
  score = knn.score(X_test, y_test)
  print("The number of neighbors is ", knn.n_neighbors,"| Accuracy is ", round(score, 4))
  #.predict_proba outputs probabilities of classes
  y_proba = knn.predict_proba(X_test)

  #calculate false positive rates and true positive rates
  fpr, tpr, _ = roc_curve(y_test, y_proba[:,1], pos_label='M')

  #calculate auc
  roc_auc = auc(fpr, tpr)

  #plot new figure
  plt.figure()

  #draw diagonal line
  plt.plot([0, 1], [0, 1], linewidth=2, linestyle='--')

  #plot roc
  plt.plot(fpr, tpr, linewidth=2, label='ROC curve (area = %0.2f)' % roc_auc)
  #scale the picture
  plt.xlim([-0.05, 1.0])
  plt.ylim([0.0, 1.05])

  #add caption
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  plt.show()
  
from sklearn.preprocessing import RobustScaler
#create new scaler
scaler = RobustScaler()

#train the scaler
X_train_scaled = scaler.fit_transform(X_train)

#apply the scaler
X_test_scaled = scaler.transform(X_test)

for i in range(1,11):
  knn = KNeighborsClassifier(n_neighbors=i)

  #train on train
  knn.fit(X_train_scaled, y_train)

  #test on test
  score = knn.score(X_test_scaled, y_test)
  print("The number of neighbors is ", knn.n_neighbors,"| Accuracy is ", round(score, 4))
  #.predict_proba outputs probabilities of classes
  y_proba = knn.predict_proba(X_test_scaled)

  #calculate false positive rates and true positive rates
  fpr, tpr, _ = roc_curve(y_test, y_proba[:,1], pos_label='M')

  #calculate auc
  roc_auc = auc(fpr, tpr)

  #plot new figure
  plt.figure()

  #draw diagonal line
  plt.plot([0, 1], [0, 1], linewidth=2, linestyle='--')

  #plot roc
  plt.plot(fpr, tpr, linewidth=2, label='ROC curve (area = %0.2f)' % roc_auc)
  #scale the picture
  plt.xlim([-0.05, 1.0])
  plt.ylim([0.0, 1.05])

  #add caption
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  plt.show()
  
  
# Spam task
  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from google.colab import files

files.upload()
df = pd.read_csv("spam.csv")
df.head()

X = df.drop("label", axis = 1).values
y = df["label"].values

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

X_test.shape

for i in range(1,11):
  knn = KNeighborsClassifier(n_neighbors=i)

  #train on train
  knn.fit(X_train, y_train)

  #test on test
  score = knn.score(X_test, y_test)
  print("The number of neighbors is ", knn.n_neighbors,"| Accuracy is ", round(score, 4))
  #.predict_proba outputs probabilities of classes
  y_proba = knn.predict_proba(X_test)

  #calculate false positive rates and true positive rates
  fpr, tpr, _ = roc_curve(y_test, y_proba[:,1])

  #calculate auc
  roc_auc = auc(fpr, tpr)

  #plot new figure
  plt.figure()

  #draw diagonal line
  plt.plot([0, 1], [0, 1], linewidth=2, linestyle='--')

  #plot roc
  plt.plot(fpr, tpr, linewidth=2, label='ROC curve (area = %0.2f)' % roc_auc)
  #scale the picture
  plt.xlim([-0.05, 1.0])
  plt.ylim([0.0, 1.05])

  #add caption
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  plt.show()
  
from sklearn.preprocessing import RobustScaler
#create new scaler
scaler = RobustScaler()

#train the scaler
X_train_scaled = scaler.fit_transform(X_train)

#apply the scaler
X_test_scaled = scaler.transform(X_test)

for i in range(1,11):
  knn = KNeighborsClassifier(n_neighbors=i)

  #train on train
  knn.fit(X_train_scaled, y_train)

  #test on test
  score = knn.score(X_test_scaled, y_test)
  print("The number of neighbors is ", knn.n_neighbors,"| Accuracy is ", round(score, 4))
  #.predict_proba outputs probabilities of classes
  y_proba = knn.predict_proba(X_test_scaled)

  #calculate false positive rates and true positive rates
  fpr, tpr, _ = roc_curve(y_test, y_proba[:,1])

  #calculate auc
  roc_auc = auc(fpr, tpr)

  #plot new figure
  plt.figure()

  #draw diagonal line
  plt.plot([0, 1], [0, 1], linewidth=2, linestyle='--')

  #plot roc
  plt.plot(fpr, tpr, linewidth=2, label='ROC curve (area = %0.2f)' % roc_auc)
  #scale the picture
  plt.xlim([-0.05, 1.0])
  plt.ylim([0.0, 1.05])

  #add caption
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  plt.show()
  
