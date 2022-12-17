#importing the necessary libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

data_all = pd.read_csv('C:/Users/sachin vinaayak/Downloads/indian_liver_patient.csv') #reading the given dataset

#data pre-processing steps
#Removing rows that contain NA values
data_all = data_all.dropna(axis=0)
data = data_all.copy()
# data  = data.drop(['Total_Protiens', 'Albumin_and_Globulin_Ratio', 'Alamine_Aminotransferase'], axis=1)

X = data.iloc[:, :-1] #contains all features except the labels column
y = data.iloc[:, -1] #contains labels i.e "1" for patient with liver disease, "2" patient with no liver disease
y = y-1

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2) #splitting the available dataset to training and testing dataset, with 20% of the dataset being testing dataset

#converting categorical feautures to binary features i.e Male & Female to 1 & 0
le = LabelEncoder()
X_train['Gender'] = le.fit_transform(X_train['Gender'])
X_val['Gender'] = le.transform(X_val['Gender'])

data_train = X_train.copy()
data_train['Dataset'] = y_train

data_train

data_train.corr()['Dataset'].abs().sort_values()#calculating the correlation between features and the dependent variable, and calculating their absolute values

# X_train_scaled = X_train.copy()
# # y_train = y_train.copy()
# X_val_scaled = X_val.copy()
# # y_val = y_val.copy()

#scaling the dataset by subtracting mean and dividing by standard deviation
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_val_scaled = sc.transform(X_val)

#using logistic regression and evaluating performance
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)#on training data
y_pred_logreg = logreg.predict(X_val_scaled)#obtaining labels on validation data
print(classification_report(y_pred_logreg, y_val))#checking performance

#using KNN with default k=5
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_val_scaled)
print(classification_report(y_pred_knn, y_val))

#using random forest with 200 trees
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_val_scaled)
print(classification_report(y_pred_rf, y_val))

#using decisiontree classifier with default values
dt = DecisionTreeClassifier()
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_val_scaled)
print(classification_report(y_pred_dt, y_val))

# Using the Random Forest Classifier due to high macro-F1 score, and performing 10 fold cross validation on validation dataset
cv = cross_val_score(rf, X_val_scaled, y_val, cv=10)
np.mean(cv)

# performing inference on new unseen sample
data_new = pd.DataFrame([[25, 'Female', 1.5, 0.1, 250, 40, 60, 6.0, 2.0, 0.9]], columns = X_val.columns)
data_new['Gender'] = le.transform(data_new['Gender'])
data_new = sc.transform(data_new)
rf.predict(data_new)

