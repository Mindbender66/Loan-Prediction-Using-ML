#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as p
import numpy as n
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import scipy.stats as ss
from scipy.stats import chi2_contingency

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


d = p.read_csv('loan_train.csv')
d


# In[91]:


d.count()


# In[92]:


d.describe()


# In[93]:


d = d.dropna()
d


# # Droping Null Values

# In[3]:


d.count()


# In[4]:


print("No. of Applicants :\n", d["Dependents"].value_counts())
print("No. of Applicants Educated or not :\n", d["Education"].value_counts())
print("No. of Applicants :\n", d["Self_Employed"].value_counts())
print("No. of Applicants :\n", d["Married"].value_counts())
print("No. of Applicants Male and female :\n", d["Gender"].value_counts())
print("No. of Applicants :\n", d["Area"].value_counts())
print("No. of Applicants :\n", d["Credit_History"].value_counts())
print("No. of Applicants :\n", d["Status"].value_counts())


# In[8]:


d.groupby("Self_Employed").min().transpose()


# # Correlation
# * Correlation for only Numeric Variables/Columns

# In[10]:


d


# In[11]:


d.corr()


# In[12]:


obs=d.pivot_table(index="Gender",columns="Dependents",values="Education",aggfunc=len).transpose().fillna(0)
chi2_contingency(obs)


# In[13]:


ful = d[["Gender","Married","Dependents","Education","Self_Employed","Applicant_Income","Coapplicant_Income","Loan_Amount","Term","Credit_History","Area","Status"]]
ful


# In[14]:


def cramers_V(v1,v2):
    crosstab = n.array(p.crosstab(v1,v2,rownames=None, colnames=None))
    stat = chi2_contingency(crosstab)[0]
    obs = n.sum(crosstab)
    mini = min(crosstab.shape)-1
    return (stat/(obs*mini))


# In[15]:


rows = []
for v1 in ful:
    col = []
    for v2 in ful:
        cramers = cramers_V(ful[v1],ful[v2])
        col.append(round(cramers,2))
    rows.append(col)
cramers_results = n.array(rows)
full_corr = p.DataFrame(cramers_results, columns = ful.columns, index = ful.columns)


# In[16]:


full_corr


# In[17]:


con = n.random.randint(low=1,high=100,size=(60,60))
annot = True
hm = sns.heatmap(data=full_corr,annot=annot)
plt.show()


# # Data Preprocessing 

# # For Train Data

# In[18]:


d = p.read_csv('loan_train.csv')
d


# In[19]:


d = d.dropna()


# In[20]:


d.head()


# # Method 1
# # To convert Categorical to Numeric 

# In[21]:


dum0 = p.get_dummies(d['Gender'])
dum0


# * Male= 1
# * Female = 0

# In[22]:


d = p.concat ((d,dum0),axis=1)
d = d.drop(['Gender'],axis=1)
d = d.drop(['Female'],axis=1)
d = d.rename(columns={"Male":"Gender"})


# In[23]:


d


# In[24]:


dum1 = p.get_dummies(d['Education'])
dum1


# * Graduate = 1
# * Not Graduate = 0

# In[25]:


d = p.concat((d,dum1),axis=1)
d = d.drop(["Not Graduate"],axis=1)
d = d.drop(["Education"],axis=1)
d = d.rename(columns={"Graduate":"Education"})


# In[26]:


d


# In[27]:


dum2 = p.get_dummies(d["Self_Employed"])
dum2


# * Yes = 1
# * No = 0

# In[28]:


d = p.concat((d,dum2),axis=1)
d = d.drop(["Self_Employed","No"],axis=1)
d = d.rename(columns={"Yes":"Self_Employed"})


# In[29]:


d


# In[30]:


dum3 = p.get_dummies(d["Married"])
dum3


# * Yes = 1
# * No = 0

# In[31]:


d = p.concat((d,dum3),axis=1)
d = d.drop(["No","Married"],axis=1)
d = d.rename(columns={"Yes":"Married"})


# In[32]:


d


# # Method 2
# # To convert Categorical to Numeric 

# In[33]:


dum4 = {
    "0":0,
    "1":1,
    "2":2,
    "3+":3
}


# * 0 = 0
# * 1 = 1
# * 2 = 2
# * 3+ = 3

# In[34]:


d["Dependents"]=d["Dependents"].map(dum4)
d


# In[35]:


dum5 = {
    "Rural":0,
    "Semiurban":1,
    "Urban":2
}


# * Rural = 0
# * Semiurban = 1
# * Urdan = 2

# In[36]:


d["Area"] = d["Area"].map(dum5)
d


# In[37]:


dum6 = p.get_dummies(d["Status"])
dum6


# In[38]:


d = p.concat((d,dum6),axis=1)
d = d.drop(["N","Status"],axis=1)
d = d.rename(columns={"Y":"Status"})


# In[39]:


d


# In[40]:


d.corr()


# In[41]:


corr = d.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# # Spliting The Data Set

# In[42]:


d


# In[43]:


d.count()


# In[44]:


x = d.iloc[:,0:11]
y = d.loc[:,"Status"]


# In[45]:


x


# In[46]:


y


# * Splitting the dataset into Training and Test Set

# In[47]:


x_train, x_test, y_train,y_test = train_test_split(x,y, test_size = 0.35, random_state= 0)


# In[48]:


sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# # Logistic Regression

# In[49]:


classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)


# In[50]:


y_pred = classifier.predict(x_test)


# In[51]:


from sklearn import metrics
LR = LogisticRegression()
LR.fit(x_train, y_train)
pred_test = LR.predict(x_test)
conf = metrics.confusion_matrix(y_test, y_pred)
print ("Confusion Matrix : \n", conf)
print(classification_report(y_test,y_pred))
print("The accuracy of Logistic Regression model is :{:.2f} ".format(100*metrics.accuracy_score(y_test, y_pred)))


# # Decision Tree

# In[52]:


dtree = DecisionTreeClassifier()
dtree.fit(x_train,y_train)


# In[53]:


predDT = dtree.predict(x_test)
print("Confusion Matrix : \n",confusion_matrix(y_test,predDT))
print(classification_report(y_test,predDT))
print("The accuracy of Decision Tree model is : ",100*metrics.accuracy_score(y_test,predDT))


# # Linear Regression 

# In[54]:


r = LinearRegression()
r.fit(x_train,y_train)


# In[55]:


y_predLR = r.predict(x_test)
print("Mean Squared Error:",metrics.mean_squared_error(y_test,y_predLR))
print("Root Mean Squared Error:", n.sqrt(metrics.mean_squared_error(y_test,y_predLR)))
print("Training set score: {:.2f}".format(r.score(x_train, y_train)))
print("Test set score: {:.2f}".format(r.score(x_test, y_test)))


# # Random Forest

# In[56]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(x_train, y_train)


# In[57]:


y_predRF = RF.predict(x_test)
con_mat = confusion_matrix(y_test, y_predRF)
print("Confusion matrix:\n",con_mat)
print(classification_report(y_test, y_predRF))
RF_SC = metrics.accuracy_score(y_predRF,y_test)
print("The accuracy of Naive Bayes Model GaussionNB is :",100*metrics.accuracy_score(y_test, y_predRF))


# # Naive Bayes Model GaussianNB

# In[58]:


from sklearn.naive_bayes import GaussianNB
nb_c = GaussianNB()
nb_c.fit(x_train, y_train)


# In[59]:


y_predNB = nb_c.predict(x_test)
con_mat = confusion_matrix(y_test, y_predNB)
print("Confusion matrix:\n",con_mat)
print(classification_report(y_test,y_predNB))
print("The accuracy of Naive Bayes Model GaussionNB is : ",100*metrics.accuracy_score(y_test, y_predNB))


# # Support Vector Machine

# In[60]:


model = SVC()
model.fit(x_train, y_train)


# In[61]:


pred_SVM = model.predict(x_test)
con_mat = confusion_matrix(y_test, pred_SVM)
print("Confusion matrix:\n",con_mat)
print(classification_report(y_test,pred_SVM))
print("The accuracy of SVM model is : ",100*metrics.accuracy_score(y_test, pred_SVM))


# # Logistic Regression Got more Accuracy 80%
