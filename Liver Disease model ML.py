from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
root = os.path.dirname(__file__)
path_df = os.path.join(root, 'indian_liver_patient.csv')
data = pd.read_csv(path_df)
data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(0.947064)
cat_cols = ['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']
X = data.iloc[:, :-1].values
y = data.iloc[:, 10].values
print(y[0:1]) 
#print(data['BP'].head()) 
#print(data['class'].head()) 
Labelx=LabelEncoder()
X[:,1]=Labelx.fit_transform(X[:,1])

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.25)
print("fffffff")
print(X_train[0:1]) 
print(Y_train[0:10])
print("test")
print(X_test[0:1])
print(Y_test[0:1])
 
clf = RandomForestClassifier()

# Training the classifier
clf.fit(X_train, Y_train)

lr=LogisticRegression()
lr.fit(X_train,Y_train)


from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix


#create a dictionary of base learners
estimators=[('rfc', clf), ('lr', lr)]
#create voting classifier
majority_voting = VotingClassifier(estimators, voting='hard')

#fit model to training data
majority_voting.fit(X_train, Y_train)
#test our model on the test data
majority_voting.score(X_test, Y_test)


# get predictions from best model above
y_preds_mv = majority_voting.predict(X_test)
print('majority voting accuracy: ',majority_voting.score(X_test, Y_test))
print('\n')

print('\n')
print(classification_report(Y_test, y_preds_mv))


# Testing model accuracy. Average is taken as test set is very small hence accuracy varies a lot everytime the model is trained
acc = 0
acc_binary = 0
for i in range(0, 20):
    Y_hat = clf.predict(X_test)
    Y_hat_bin = Y_hat>0
    Y_test_bin = Y_test>0
    acc = acc + accuracy_score(Y_hat, Y_test)
    acc_binary = acc_binary +accuracy_score(Y_hat_bin, Y_test_bin)

print("Average test Accuracy:{}".format(acc/20))
print("Average binary accuracy:{}".format(acc_binary/20))

# Saving the trained model for inference
model_path = os.path.join(root, 'liVERFCMODEL.sav')
joblib.dump(clf, model_path)
model_path1 = os.path.join(root, 'LIVERVOTINGMODEL.sav')
joblib.dump(majority_voting, model_path1)
# Saving the scaler object
scaler_path = os.path.join(root, 'LIVERscaler.pkl')
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(Labelx, scaler_file)

scaler_path = os.path.join(os.path.dirname(__file__), 'LIVERscaler.pkl')
scaler = None
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

a=65
b="Female"
c=0.7
d=0.1
e=187
g=16
h=18	
i=6.8	
j=3.3	
k=0.9	

vector = np.vectorize(np.float)
check = np.array([a,b,c,d,e,g,h,i,j,k]).reshape(1, -1)

Labe=LabelEncoder()
check[:,1]=Labe.fit_transform(check[:,1])
model_path = os.path.join(os.path.dirname(__file__), 'liVERFCMODEL.sav')




 
check = vector(check)
#print(check) 

print(X_test[0:1])
print(check[[0]] )
clf = joblib.load(model_path)
B_pred = clf.predict(check[[0]])
if B_pred == 2:
    print("lIVER DISEASE DETECTED")
if B_pred == 1:
    print("NO DISEASE DETECTED")