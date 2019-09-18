import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('/home/pratik/Downloads/fraud_detection/creditcard.csv',header=0)

y=df.Class
X=df.drop('Class',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)


dummy=DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
dummy_pred=dummy.predict(X_test)

print('unique predicted label:',np.unique(dummy_pred))
print('accuracy:',accuracy_score(y_test,dummy_pred))

lr=LogisticRegression(class_weight='balanced',solver='liblinear').fit(X_train,y_train)

lr_pred=lr.predict(X_test)
accuracy_score(y_test,lr_pred)

predictions=pd.DataFrame(lr_pred)
predictions[0].value_counts()

f1_score(y_test,lr_pred)
recall_score(y_test,lr_pred)

rfc=RandomForestClassifier(n_estimators=500).fit(X_train,y_train)

rf_pred=rfc.predict(X_test)
accuracy_score(y_test,rf_pred)

f1_score(y_test,rf_pred)
recall_score(y_test,rf_pred)

from sklearn.utils import resample
 
y=df.Class
X=df.drop('Class',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

X=pd.concat([X_train,y_train],axis=1)

not_fraud=X[X.Class==0]
fraud=X[X.Class==1]

fraud_upsampled=resample(fraud,
                         replace=True,# sample with replacement
                         n_samples=len(not_fraud),# match number in majority class
                         random_state=27)

upsampled=pd.concat([not_fraud,fraud_upsampled])
upsampled.Class.value_counts()

y_train=upsampled.Class
X_train=upsampled.drop('Class',axis=1)

lr_upsampled=LogisticRegression(class_weight='balanced',solver='liblinear').fit(X_train,y_train)
upsampled_pred=lr_upsampled.predict(X_test)

accuracy_score(y_test,upsampled_pred)
f1_score(y_test,upsampled_pred)
recall_score(y_test,upsampled_pred)

not_fraud_undersampled=resample(not_fraud,
                               replace=False,
                               n_samples=len(fraud),
                               random_state=42)

undersampled=pd.concat([fraud,not_fraud_undersampled])
undersampled.Class.value_counts()

y_train=undersampled.Class
X_train=undersampled.drop('Class',axis=1)

lr_undersampled=LogisticRegression(class_weight='balanced',solver='liblinear').fit(X_train,y_train)
undersampled_pred=lr_undersampled.predict(X_test)

accuracy_score(y_test,undersampled_pred)
f1_score(y_test,undersampled_pred)
recall_score(y_test,undersampled_pred)

from imblearn.over_sampling import SMOTE

y=df.Class
X=df.drop('Class',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

sm=SMOTE(random_state=42,ratio=1)
X_train,y_train=sm.fit_sample(X_train,y_train)

smote_lr=LogisticRegression(class_weight='balanced',solver='liblinear').fit(X_train,y_train)
smote_pred=smote_lr.predict(X_test)

accuracy_score(y_test,smote_pred)
f1_score(y_test,smote_pred)
recall_score(y_test,smote_pred)


print('f1_score of logistic_regression with balanced parameter:',f1_score(y_test,lr_pred))
print('recall_value of logistic_regression with balanced parameter:',recall_score(y_test,lr_pred))


print('f1_score of random_forest:',f1_score(y_test,rf_pred))
print('recall_value of random_forest:',recall_score(y_test,rf_pred))


print('f1_score of upsampling:',f1_score(y_test,upsampled_pred))
print('recall_value of upsampling:',recall_score(y_test,upsampled_pred))


print('f1_score of undersampling:',f1_score(y_test,undersampled_pred))
print('recall_value of undersampling:',recall_score(y_test,undersampled_pred))


print('f1_score of SMOTE:',f1_score(y_test,smote_pred))
print('recall_value of undersamplin:',recall_score(y_test,smote_pred))
