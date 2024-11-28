# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: AKASH CT
RegisterNumber:  24901150
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
```
```
print("data.info():")
data.info()
```
```
print("isnull() and sum():")
data.isnull().sum()
```
```
print("data value counts():")
data["left"].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
```
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
```
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
```
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```

## Output:
![Screenshot 2024-11-28 215904](https://github.com/user-attachments/assets/86a4a80b-5141-46a9-b468-2a90b399d751)
![Screenshot 2024-11-28 215928](https://github.com/user-attachments/assets/b5d387f9-0ae0-48b7-9a73-1ee316d22e7a)
![Screenshot 2024-11-28 215939](https://github.com/user-attachments/assets/26c838e0-a45b-4ed6-b355-4415c0d2ab50)
![Screenshot 2024-11-28 215948](https://github.com/user-attachments/assets/90293165-203b-46a5-956b-578a7423d71c)
![Screenshot 2024-11-28 220001](https://github.com/user-attachments/assets/be473314-4f54-40d9-9e31-a93d0d9fdc3c)
![Screenshot 2024-11-28 220009](https://github.com/user-attachments/assets/4b3bca6f-afe8-414f-969f-96adb5147272)
![Screenshot 2024-11-28 220017](https://github.com/user-attachments/assets/fe94b9dc-96a4-42dc-a569-a0ca49ea02a2)
![Screenshot 2024-11-28 220035](https://github.com/user-attachments/assets/46587e95-7e45-4f37-b76a-436fae7131e8)
![Screenshot 2024-11-28 220045](https://github.com/user-attachments/assets/2879fa1a-d434-4eee-ae34-4594ab0e5ffa)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
