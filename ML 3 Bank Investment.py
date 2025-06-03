#!/usr/bin/env python
# coding: utf-8

# In[1]:


#for data preparation and analysis
import pandas as pd
#for creating plots
import matplotlib.pyplot as plt
#for distribution plot and heatmap
import seaborn as sns

#for creating training and test samples
from sklearn.model_selection import train_test_split

#feature selection (to select significant variables)
from sklearn.feature_selection import SelectKBest, f_regression

#for building Linear Regression model
from sklearn.linear_model import LinearRegression


# In[2]:


df=pd.read_csv(r"C:\Users\amith\OneDrive\Desktop\introtallent\python\Data Files used in Projects\Data Files used in Projects\Investment.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


df.describe()


# In[9]:


df.columns


# # Variable Description
# * Target Variable: "Invested"
# * Independant variables:
# * Age
# * Job
# * Marital
# * Education
# * Default
# * Housing
# * Loan
# * Contact
# * Month
# * Day_of_week
# * Duration
# * Campaign
# * Pdays
# * Previous
# * Poutcome
# * Emp_var_rate
# * cons_price_idx
# * cons_conf_idx
# * euribor3m
# * nr_employed

# In[10]:


df.isnull().sum()


# In[11]:


# Check Outliers


# In[12]:


plt.boxplot(df["age"])
plt.show()


# In[13]:


plt.boxplot(df["age"])
plt.show()


# In[14]:


plt.boxplot(df["campaign"])
plt.show()


# In[15]:


plt.boxplot(df["pdays"])
plt.show()


# In[16]:


plt.boxplot(df["previous"])
plt.show()


# In[17]:


plt.boxplot(df["emp_var_rate"])
plt.show()


# In[18]:


plt.boxplot(df["cons_price_idx"])
plt.show()


# In[19]:


plt.boxplot(df["cons_conf_idx"])
plt.show()


# In[20]:


plt.boxplot(df["euribor3m"])
plt.show()


# In[21]:


plt.boxplot(df["nr_employed"])
plt.show()


# In[23]:


#user defined function outlier treatment

def remove_outlier(d,c):
    #find q1 and q3
    q1=d[c].quantile(0.25)
    q3=d[c].quantile(0.75)
    
    #calculate iqr
    iqr=q3-q1
    
    #calculate upper bound (ub) and lower bound(lb)
    ub=q3+1.5*iqr
    lb=q1-1.5*iqr
    
    #select good data and exclude outlier
    good_data=d[(d[c]>lb) & (d[c]<ub)]
    
    return good_data


# In[24]:


df=remove_outlier(df,"age")
plt.boxplot(df["age"])
plt.show()


# In[30]:


df=remove_outlier(df,"duration")
plt.boxplot(df["duration"])
plt.show()


# In[31]:


df=remove_outlier(df,"cons_conf_idx")
plt.boxplot(df["cons_conf_idx"])
plt.show()


# In[32]:


df.shape


# # EDA (Exploratory Data Analysis)
# * Distribution
# * Data Mix
# * Corelation

# In[34]:


df.columns


# In[35]:


# "age", "duration, "emp_var_rate", "cons_price_idx", "euribor3m", "nr_employed"


# In[36]:


sns.distplot(df["age"])


# In[37]:


sns.distplot(df["duration"])


# In[38]:


sns.distplot(df["emp_var_rate"])


# In[39]:


sns.distplot(df["cons_price_idx"])


# In[40]:


sns.distplot(df["euribor3m"])


# In[41]:


sns.distplot(df["nr_employed"])


# In[42]:


sns.distplot(df["campaign"])


# In[43]:


sns.distplot(df["pdays"])


# In[44]:


sns.distplot(df["previous"])


# In[45]:


df.columns


# In[46]:


#check datamix


# In[47]:


df.groupby('job')['job'].count().plot(kind='bar')


# In[48]:


df.groupby('marital')['marital'].count().plot(kind='bar')


# In[50]:


df.groupby('education')['education'].count().plot(kind='bar')


# In[51]:


df.groupby('default')['default'].count().plot(kind='bar')


# In[52]:


df.groupby('housing')['housing'].count().plot(kind='bar')


# In[53]:


df.groupby('loan')['loan'].count().plot(kind='bar')


# In[54]:


df.groupby('contact')['contact'].count().plot(kind='bar')


# In[55]:


df.groupby('month')['month'].count().plot(kind='bar')


# In[56]:


df.groupby('day_of_week')['day_of_week'].count().plot(kind='bar')


# In[57]:


df.groupby('poutcome')['poutcome'].count().plot(kind='bar')


# In[58]:


df.groupby('Invested')['Invested'].count().plot(kind='bar')


# # Pearson Correlation

# In[60]:


#create a set of numeric columns
df_numeric=df.select_dtypes(include=['int64','float64'])
df_numeric.head()


# In[61]:


df['job'].unique()


# In[62]:


df['marital'].unique()


# In[63]:


df['education'].unique()


# In[64]:


df['default'].unique()


# In[65]:


df['housing'].unique()


# In[66]:


df['loan'].unique()


# In[ ]:




