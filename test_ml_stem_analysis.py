#importing necessary libraries
import pytest
import numpy as np
from sklearn import preprocessing, cross_validation
import pandas as pd
import xgboost
from xgboost import XGBClassifier 



#reading data file
data = pd.read_excel('https://query.data.world/s/swx6vzqc5vwjo4mhia4sxylvzgd6pq')
#naming important columns
data.rename(columns={'Unnamed: 5':'Fund2008','FUNDING':'Years','Unnamed: 6':'Fund2009','Unnamed: 7':'Fund2010'},inplace=True)

data['Fund2008'] = data.iloc[1:,5]
data['Fund2009'] = data.iloc[1:,6]
data['Fund2010'] = data.iloc[1:,7]
#Filling n/a values with mean
data.fillna(value = 0,inplace=True)
data.iloc[1:,5].replace(0,data["Fund2008"].mean(),inplace=True)
data.iloc[1:,6].replace(0,data["Fund2009"].mean(),inplace=True)
data.iloc[1:,7].replace(0,data["Fund2010"].mean(),inplace=True)

#selection of features
X = data[['Fund2008','Fund2009','Fund2010']]
y = data[['Unnamed: 8']]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
model = XGBClassifier()
model.fit(X_train, y_train)

#Funtion to test the test cases using pytest
#here we can take raw input from user as well
#and check the results
#I am uploading screenshot of how to use the pytest framework and result of my code!!!
def test_ml_stem_analysis():
    df = pd.DataFrame(columns=["Fund2008", "Fund2009", "Fund2010"])
    dp = 34
    st = 45
    et = 56
    df1 = pd.DataFrame(data=[[dp,st,et]],columns=["Fund2008", "Fund2009", "Fund2010"])
    df = pd.concat([df,df1], axis=0)
    df.index = range(len(df.index))
    df = df.astype(float)
    u_pred = model.predict(df)
    return u_pred
    assert  u_pred == [3]
