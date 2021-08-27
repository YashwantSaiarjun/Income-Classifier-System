import os
os.chdir('C:\\Users\\indra\\Documents\\DataScience_ML\\csvfiles')

import pandas as pd
import numpy as np
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix


data2 = pd.read_csv('income.csv')

data=data2.copy(deep=True)

print(data.info())
 
print(data.isnull().sum())

print(data.describe())

print(data.describe(include="O"))


print(data['JobType'].value_counts())
print(data['EdType'].value_counts())
print(data['maritalstatus'].value_counts())
print(data['occupation'].value_counts)
print(data['race'].value_counts())
print(data['gender'].value_counts())
print(data['nativecountry'].value_counts())
print(data['SalStat'].value_counts())



df=pd.read_csv('income.csv',na_values=[" ?"])

missing=df[df.isnull().any(axis=1)]
print(df['JobType'].value_counts())

df=df.dropna(axis=0)

corelation=df.corr()

print(df.columns)

print(pd.crosstab(index=df['gender'],columns='count',normalize=True))

print(pd.crosstab(index=df['gender'],columns=data['SalStat'],normalize='index'))



sns.countplot(x=df['SalStat'])
sns.distplot(x=df['age'],bins=10,kde=True)
sns.boxplot(x=df['SalStat'],y=df['age'])

print(pd.crosstab(index=df['JobType'],columns=df['SalStat'],normalize='index'))

sns.countplot(y=df['JobType'],hue=data['SalStat'])

print(pd.crosstab(index=df['EdType'],columns=df['SalStat'],normalize=True))
sns.countplot(y=df['EdType'],hue=data['SalStat'])

print(pd.crosstab(index=df['occupation'],columns=df['SalStat'],normalize=True))
sns.countplot(y=df['occupation'],hue=df['SalStat'])

sns.boxplot(y=df['hoursperweek'],x=df['SalStat'])



