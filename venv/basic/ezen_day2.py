import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ctx = 'C:/Users/ezen/PycharmProjects/test1/titanic/data/'
train = pd.read_csv(ctx+'train.csv')
test = pd.read_csv(ctx+'test.csv')

df = pd.DataFrame(train)
print(df.columns)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))
    plt.show()




train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
train.head()
test.head()


s_city = train[train["Embarked"]=='S'].shape[0]
print("S :", s_city) # S : 646
c_city = train[train["Embarked"]=='C'].shape[0]
print("C :", c_city) # C : 168
q_city = train[train["Embarked"]=='Q'].shape[0]
print("Q :", q_city) # Q : 77


train = train.fillna({"Embarked":"S"})

city_mapping = {"S":1, "C":2, "Q":3}
train['Embarked'] = train['Embarked'].map(city_mapping)
test['Embarked'] = test['Embarked'].map(city_mapping)
print(train.head())
print(test.head())


combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'])





combine=[train, test]
for dataset in combine:
    dataset['Title' ]=dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
print(pd.crosstab(train['Title'],train['Sex']))


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())



train = train.drop(['Name','PassengerId'], axis = 1)
test = test.drop(['Name'], axis = 1)
combine = [train, test]
print(train.head())


sex_mapping = {"male":0, "female":1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
print(train.head())


import numpy as np

train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels = labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels = labels)
print(train.head())


title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5,'Rare':6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
print(train.head())

age_title_mapping = {0:"Unknown", 1: "Young Adult", 2:"Student", 3:"Adult", 4:"Baby", 5:"Adult", 6:"Adult"}
for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train['Title'][x]]
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test['Title'][x]]
print(train.head())


age_mapping = {'Baby' : 1, 'Child':2, 'Teenager':3, 'Student': 4, 'Young Adult':5,'Adult':6, 'Senior':7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = train['AgeGroup'].map(age_mapping)
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)
print(train.head())




train['FareBand'] = pd.qcut(train['Fare'], 4, labels = {1,2,3,4})
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = {1,2,3,4})


train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)
print(train.head())



train_data = train.drop('Survived', axis = 1)
target = train['Survived']
print(train_data.shape)
print(target.shape)

train.info()