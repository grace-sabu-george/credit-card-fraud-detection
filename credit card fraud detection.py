#!/usr/bin/env python
# coding: utf-8

# # ML APPROACH 

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")


# #  Loading Dataset

# In[2]:


data=pd.read_csv("D:\project 2021-2023\creditcard.csv")


# # EDA

# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


#checking the shape
data.shape


# In[6]:


#checking the datatypes
data.info()


# In[7]:


#checking for null values
data.isnull().sum()


# In[8]:


#checking distribution of numerical values 
data.describe().T


# #  Data Visualization

# In[9]:


# Checking for the Outliers in the dataset
plt.figure(figsize = (15,25))
count = 1
for col in data:
    plt.subplot(11,3,count)
    plt.boxplot(data[col])
    plt.title(col)
    count += 1
plt.show()


# In[10]:


#checking the class distribution
data['Class'].value_counts()


# In[11]:


#dropping the coloumn
data=data.drop(["Unnamed: 0"],axis=1)
data


# In[12]:


#checking the class distribution of the target variable in percentage and plotting piechart
print((data.groupby('Class')['Class'].count()/data['Class'].count())*100)
((data.groupby('Class')['Class'].count()/data['Class'].count())*100).plot.pie()


# In[13]:


#creating a barplot for the number and percentagge of fraudulent vs non fraudulent
plt.figure(figsize=(7,5))
sns.countplot(data['Class'])
plt.title("Class Count",fontsize=15)
plt.xlabel("Record counts by class",fontsize=12)
plt.ylabel("Count",fontsize=12)
plt.show()


# In[14]:


#DistPlot
sns.distplot(data['Class'])
plt.title ("Class Count")


# In[15]:


#checking the correlation
corr=data.corr()
corr


# In[16]:


#checking the correlation in heatmap
plt.figure(figsize=(24,18))
sns.heatmap(corr,cmap="coolwarm",annot=True)
plt.show()


# In[17]:


#Dropping the column
data.drop(['Time'],axis=1,inplace=True)
data.head()


# # Splitting the data 

# In[18]:


#splitting for train and test 
x=data.drop(['Class'],axis=1)
y=data['Class']


# In[19]:


x.head()


# In[20]:


y.head()


# In[21]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[22]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[23]:


print(np.sum(y))
print(np.sum(y_train))
print(np.sum(y_test))


# In[24]:


#accumulating all coloumn names under one variable
cols = list(x.columns.values)


# # Model Building

# ## 1.logistic Regression

# In[25]:


#importing logistic libraries and fitting the model
from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model.fit(x_train,y_train)


# In[26]:


# Predictions on training and testing data
y_pred_train1 = log_model.predict(x_train)
y_pred_test1 = log_model.predict(x_test)


# In[27]:


#checking the accuracy of the model
accu_train1=accuracy_score(y_train,y_pred_train1)
print(accu_train1)
accu_test1=accuracy_score(y_test,y_pred_test1)
print(accu_test1)


# In[28]:


print("training accuracy:",round(accu_train1*100,2),"%")
print("testing accuracy:",round(accu_test1*100,2),"%")


# In[29]:


#confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test1).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix


# In[30]:


sns.heatmap(conf_matrix, annot=True)


# In[31]:


#checking precision,recall,f1-score
print(classification_report(y_test,y_pred_test1))


# ## 2.Random Forest

# In[32]:


#importing Random Forest libraries and fitting the model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)


# In[33]:


# Predictions on training and testing data
y_pred_train2 = rf_model.predict(x_train)
y_pred_test2 = rf_model.predict(x_test)


# In[34]:


#checking the accuracy of the model
accu_train2=accuracy_score(y_train,y_pred_train2)
print(accu_train2)
accu_test2=accuracy_score(y_test,y_pred_test2)
print(accu_test2)


# In[35]:


print("training accuracy:",round(accu_train2*100,2),"%")
print("testing accuracy:",round(accu_test2*100,2),"%")


# In[36]:


#confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test2).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix


# In[37]:


sns.heatmap(conf_matrix, annot=True)


# In[38]:


#checking precision,recall,f1-score
print(classification_report(y_test,y_pred_test2))


# ## 3. SVM

# In[39]:


#importing SVM libraries and fitting the model
from sklearn.svm import SVC
svc_model=SVC(kernel='linear',random_state=42,probability=True)
svc_model.fit(x_train,y_train)


# In[40]:


# Predictions on training and testing data
y_pred_train3 = svc_model.predict(x_train)
y_pred_test3 = svc_model.predict(x_test)


# In[41]:


##checking the accuracy of the model
accu_train3=accuracy_score(y_train,y_pred_train3)
print(accu_train3)
accu_test3=accuracy_score(y_test,y_pred_test3)
print(accu_test3)


# In[42]:


print("training accuracy:",round(accu_train3*100,2),"%")
print("testing accuracy:",round(accu_test3*100,2),"%")


# In[43]:


#confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test3).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix


# In[44]:


sns.heatmap(conf_matrix, annot=True)


# In[45]:


#checking precision,recall,f1-score
print(classification_report(y_test,y_pred_test3))


# ## 4. KNN 

# In[46]:


#importing KNN libraries and fitting the model
from sklearn import neighbors
import sklearn.model_selection as ms


# In[47]:


K=list(range(1,21,2))
K


# In[48]:



n_grid=[{"n_neighbors":K}]#to find the number of neighbours
model=neighbors.KNeighborsClassifier()
gd=ms.GridSearchCV(estimator=model,param_grid=n_grid,cv=ms.KFold(n_splits=10))
gd.fit(x_train,y_train)


# In[49]:


best_K=gd.best_params_["n_neighbors"]
print("best value o k :",best_K)


# In[50]:


knn_model=neighbors.KNeighborsClassifier(n_neighbors=best_K)
knn_model.fit(x_train,y_train)


# In[51]:


# Predictions on training and testing data
y_pred_train4 = knn_model.predict(x_train)
y_pred_test4 = knn_model.predict(x_test)


# In[52]:


accu_train4=accuracy_score(y_train,y_pred_train4)
print(accu_train4)
accu_test4=accuracy_score(y_test,y_pred_test4)
print(accu_test4)


# In[53]:


print("training accuracy:",round(accu_train4*100,2),"%")
print("testing accuracy:",round(accu_test4*100,2),"%")


# In[54]:


#confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test4).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix


# In[55]:


sns.heatmap(conf_matrix, annot=True)


# In[56]:


#checking precision,recall,f1-score
print(classification_report(y_test,y_pred_test4))


# ## 5. Decision Tree

# In[57]:


#importing Decision Tree libraries and fitting the model
from sklearn.tree import DecisionTreeClassifier
dt_model= DecisionTreeClassifier()
dt_model.fit(x_train,y_train)


# In[58]:


# Predictions on training and testing data
y_pred_train5 = dt_model.predict(x_train)
y_pred_test5 = dt_model.predict(x_test)


# In[59]:


#checking the accuracy of the model
accu_train5=accuracy_score(y_train,y_pred_train5)
print(accu_train5)
accu_test5=accuracy_score(y_test,y_pred_test5)
print(accu_test5)


# In[60]:


print("training accuracy:",round(accu_train5*100,2),"%")
print("testing accuracy:",round(accu_test5*100,2),"%")


# In[61]:


#confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test5).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix


# In[62]:


sns.heatmap(conf_matrix, annot=True)


# In[63]:


#checking precision,recall,f1-score
print(classification_report(y_test,y_pred_test5))


# ## 6. ADA_BOOST_CLASSIFIER

# In[64]:


#importing ADA_BOOST libraries and fitting the model
from sklearn.ensemble import AdaBoostClassifier
ada_model=AdaBoostClassifier()
ada_model.fit(x_train,y_train)


# In[65]:


# Predictions on training and testing data
y_pred_train6 = ada_model.predict(x_train)
y_pred_test6 = ada_model.predict(x_test)


# In[66]:


#checking the accuracy of the model
accu_train6=accuracy_score(y_train,y_pred_train6)
print(accu_train6)
accu_test6=accuracy_score(y_test,y_pred_test6)
print(accu_test6)


# In[67]:


print("training accuracy:",round(accu_train6*100,2),"%")
print("testing accuracy:",round(accu_test6*100,2),"%")


# In[68]:


#confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test6).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix


# In[69]:


sns.heatmap(conf_matrix, annot=True)


# In[70]:


#checking precision,recall,f1-score
print(classification_report(y_test,y_pred_test6))


# ## 7. GaussianNB

# In[71]:


#importing GaussianNB libraries and fitting the model
from sklearn.naive_bayes import GaussianNB
gauss_model=GaussianNB()


# In[72]:


gauss_model.fit(x_train,y_train)


# In[73]:


# Predictions on training and testing data
y_pred_train7 = gauss_model.predict(x_train)
y_pred_test7 = gauss_model.predict(x_test)


# In[74]:


#checking the accuracy of the model
accu_train7=accuracy_score(y_train,y_pred_train7)
print(accu_train7)
accu_test7=accuracy_score(y_test,y_pred_test7)
print(accu_test7)


# In[75]:


print("training accuracy:",round(accu_train7*100,2),"%")
print("testing accuracy:",round(accu_test7*100,2),"%")


# In[76]:


#confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test7).ravel()
conf_matrix = pd.DataFrame(
    {
        'Predicted Fraud': [tp, fp],
        'Predicted Not Fraud': [fn, tn]
    }, index=['Fraud', 'Not Fraud'])
conf_matrix


# In[77]:


sns.heatmap(conf_matrix, annot=True)


# In[78]:


#checking precision,recall,f1-score
print(classification_report(y_test,y_pred_test7))


# ## Performance Analysis of each model

# In[79]:


New_model=[log_model,rf_model,svc_model,knn_model,dt_model,ada_model,gauss_model]
Testing_Acc=[]
Training_Acc=[]
for i in New_model:
  i.fit(x_train,y_train)
  Testing_Acc.append(round(accuracy_score(y_test,i.predict(x_test))*100,2))
  Training_Acc.append(round(accuracy_score(y_train,i.predict(x_train))*100,2))
print(Testing_Acc)
print(Training_Acc)  


# In[80]:


DATA_FRAME=pd.DataFrame({"MODEL_NAME":['Logistic Regression ','RandomForest','SVM','KNN','Decision Tree Classifier','AdaBoostClassifier','GaussianNB'],"training accuracy":Training_Acc,"testing accuracy":Testing_Acc})
DATA_FRAME


# # Graphical Representation of Each Model Performance using its train & test data

# In[81]:


DATA_FRAME.plot(kind='bar',figsize=(15,8),xlabel=DATA_FRAME['MODEL_NAME'])


# # Prediction using Existing Data

# In[82]:


d_data=[[-3.240186587,2.97812179,-4.162313937,3.869124379,-3.645256455,-0.126270561,-4.744729737,-0.065331044,-2.168366095,-4.758304075,3.471097846,-6.533106769,-0.98346867,-6.073989003,1.125406709,-7.718042405,-9.855927437,-5.193907893,2.042697688,-0.224043463,2.601441029,0.231910116,-0.036489849,0.042639645,-0.438330346,-0.12582108,0.421299802,0.003145876,172.32
]]
p=ada_model.predict(d_data)
print(p)
if p==0:
    print("Credit Card Fraud Detection has not been detected")
else:
    print("Credit Crad Fraud Detection has been detected")


# In[83]:


d_data=[[1.328286973,-0.579219059,0.622190545,-0.477965418,-1.062418354,-0.570816358,-0.83457999,-0.022299183,0.821265832,0.248291354,0.63104363,-4.078436048,-0.294213807,1.731927621,0.562439079,1.015576868,1.348802151,-1.494219178,0.054999048,-0.102332115,-0.230902625,-0.650906248,0.185310855,-0.057384073,0.05598644,-0.48551575,-0.017463823,0.011740384,27.91
]]
p=ada_model.predict(d_data)
print(p)
if p==0:
    print("Credit Card Fraud Detection has not been detected")
else:
    print("Credit Crad Fraud Detection has been detected")


# #  DL APPROACH

# # ANN-MLP

# In[84]:


pip install keras


# In[85]:


pip install tensorflow 


# ###  Importing Libraries for Keras

# In[86]:


from tensorflow.keras import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import plot_model


# ### Model Creation

# In[88]:


#define model
model = Sequential([
    #define input layer
    Dense(units=128,input_dim = 29,activation='relu'),
    #define hidden layer
    Dense(units=64,activation='relu'),
    Dropout(0.45),
    Dense(32,activation='relu'),
    #define output layer
    Dense(1,activation='sigmoid')
])


# In[89]:


model.summary()


# # Configuration Step

# In[90]:


#define loss and optimizer
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# ## Data Training

# In[91]:


model.fit(x_train,y_train,batch_size=128,epochs=25)


# ### Evaluation

# In[92]:


model.evaluate(x_test,y_test)


# ## Prediction Using Test data

# In[93]:


pred = model.predict(x_test)


# In[94]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print('Accuracy score is: ',round(accuracy_score(y_test,pred.round()),4))
print("============================================")
print('Classification report: \n',classification_report(y_test,pred.round()))
print("============================================")

print('Confusion Matrix: \n',confusion_matrix(y_test,pred.round()))


# # Prediction Using Existing data

# In[95]:


y_pred = model.predict([[-3.240186587,2.97812179,-4.162313937,3.869124379,-3.645256455,-0.126270561,-4.744729737,-0.065331044,-2.168366095,-4.758304075,3.471097846,-6.533106769,-0.98346867,-6.073989003,1.125406709,-7.718042405,-9.855927437,-5.193907893,2.042697688,-0.224043463,2.601441029,0.231910116,-0.036489849,0.042639645,-0.438330346,-0.12582108,0.421299802,0.003145876,172.32
]])
print(y_pred)
if p >= 0.5:
    print("Credit Card Fraud Detection has not been detected")
else:
    print("Credit Crad Fraud Detection has been detected")


# In[ ]:





# In[ ]:




