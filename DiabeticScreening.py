import pandas as pd

data = pd.read_csv('/Users/ismailsa/DATASET/Python/diabetes.csv')

#display 5 top data
data.head()

#display 5 tail data
data.tail()

#find the shape of the data
data.shape
print("Number of Rows", data.shape[0])
print("Number of Columns", data.shape[1])

#get informaton of dataset
data.info()

#check null values if any in the dataset
data.isnull()
data.isnull().sum()

#overall statistics
data.describe()

#cleaning the abnormal (e.g. BMI =0? convert to mean)
import numpy as np
data_copy = data.copy(deep=True)
data.columns
data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI']] = data_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
              'BMI']].replace(0,np.nan)
data_copy
data_copy.isnull().sum()

#consider the zero should be change the mean/ median values
data['Glucose'] = data['Glucose'].replace(0,data['Glucose'].mean())
data['BloodPressure'] = data['BloodPressure'].replace(0,data['BloodPressure'].mean())
data['SkinThickness'] = data['SkinThickness'].replace(0,data['SkinThickness'].mean())
data['Insulin'] = data['Insulin'].replace(0,data['Insulin'].mean())
data['BMI'] = data['BMI'].replace(0,data['BMI'].mean())

#store feature matrix as X and Y, independent and dependent
x = data.drop('Outcome', axis=1)
y = data['Outcome']
x

#Split the data into the testing and training to evaluate the performance
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#pipeline sklearn (change as input into the next) - 
#created the sequence in step in ML, automate workflow

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#above is the distance related, below is the classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline

#change the pipeline into different set
pipeline_lr = Pipeline([('scalar1', StandardScaler()),('lr_classifier', LogisticRegression())])
pipeline_knn = Pipeline([('scalar2', StandardScaler()),('knn_classifier', KNeighborsClassifier())])
pipeline_svc = Pipeline([('scalar3', StandardScaler()),('svc_classifier', SVC())])

#see no need standard scaler because no distance involve, below is the classification
pipeline_dt = Pipeline([('dt_classifier', DecisionTreeClassifier())])
pipeline_rf = Pipeline([('rf_classifier', RandomForestClassifier(max_depth=3))])
pipeline_gbc = Pipeline([('gbc_classifier', GradientBoostingClassifier())])

Pipeline = [pipeline_lr,pipeline_knn,pipeline_svc,pipeline_dt,pipeline_rf,pipeline_gbc]
for pipe in Pipeline:
    pipe.fit(x_train,y_train)

#Accuracy of our model
pipe_dict={0:'LR',1:'KNN',2:'SVC',3:'DT',4:'RF',5:'GBC'}
pipe_dict

#Evaluation of model:
for i, model in enumerate(Pipeline):
    print("{} Test Accuracy:{}".format(pipe_dict[i],model.score(x_test,y_test)))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
x_f = data.drop('Outcome', axis=1)
y_f = data['Outcome']
rf = RandomForestClassifier(max_depth=3)
rf.fit(x_f,y_f)

#Prediction of data/ new data
new_data = pd.DataFrame({'Pregnancies':6, 'Glucose':148.0,'BloodPressure':72.0,
                         'SkinThickness':35.0,'Insulin':79.80,'BMI':33.6,
                         'DiabetesPedigreeFunction':0.627,'Age':50,
                         },index=[0])

#predict
p = rf.predict(new_data)
if p[0]==0:
    print('non-diabetic')
else:
    print('diabetic')

#save the model using Joblib
import joblib
joblib.dump(rf,'/Users/ismailsa/DATASET/SpyderTutorial/model_joblib_diabetes')
model = joblib.load('/Users/ismailsa/DATASET/SpyderTutorial/model_joblib_diabetes')
model.predict(new_data)

#Create GUI
from tkinter import *
import joblib

def show_entry_fields():
    p1 = float(e1.get())
    p2 = float(e2.get())
    p3 = float(e3.get())
    p4 = float(e4.get())
    p5 = float(e5.get())
    p6 = float(e6.get())
    p7 = float(e7.get())
    p8 = float(e8.get())
    
    model = joblib.load('/Users/ismailsa/DATASET/SpyderTutorial/model_joblib_diabetes')
    result = model.predict([[p1, p2, p3, p4, p5, p6, p7, p8]])
    
    if result == 0:
        result_label.config(text="Non-Diabetic")
    else:
        result_label.config(text="Diabetic")

def reset_fields():
    e1.delete(0, END)
    e2.delete(0, END)
    e3.delete(0, END)
    e4.delete(0, END)
    e5.delete(0, END)
    e6.delete(0, END)
    e7.delete(0, END)
    e8.delete(0, END)
    result_label.config(text="")

master = Tk()
master.title("Diabetes Prediction Using Machine Learning")

label = Label(master, text="Diabetes Prediction Using Machine Learning", bg="black", fg="white")
label.grid(row=0, columnspan=2)

Label(master, text="Pregnancies").grid(row=1)
Label(master, text="Glucose").grid(row=2)
Label(master, text="Enter Value of Blood Pressure").grid(row=3)
Label(master, text="Enter Value of Skin Thickness").grid(row=4)
Label(master, text="Enter Value of Insulin").grid(row=5)
Label(master, text="Enter Value of BMI").grid(row=6)
Label(master, text="Enter Value of Diabetes Pedigree Function").grid(row=7)
Label(master, text="Enter Value of Age").grid(row=8)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)

predict_button = Button(master, text='Predict', command=show_entry_fields)
predict_button.grid(row=9, column=0)

reset_button = Button(master, text='Reset', command=reset_fields)
reset_button.grid(row=9, column=1)

result_label = Label(master, text="")
result_label.grid(row=10, columnspan=2)

if __name__ == '__main__':
    mainloop()
