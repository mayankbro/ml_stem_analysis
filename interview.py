#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:54:52 2018

@author: mayank
"""
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn import preprocessing, cross_validation
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


file = r'data.xls'
data = pd.read_excel(file)
data.rename(columns={'Unnamed: 5':'Fund2008','FUNDING':'Years','Unnamed: 6':'Fund2009','Unnamed: 7':'Fund2010'},inplace=True)
data.convert_objects(convert_numeric=True)

data['Fund2008'] = data.iloc[1:,5]
data['Fund2009'] = data.iloc[1:,6]
data['Fund2010'] = data.iloc[1:,7]

data.fillna(1,inplace=True)
data.iloc[1:,5].replace(1,data['Fund2008'].mean(),inplace=True)
print(data['Fund2008'])

#data.apply(lambda x: data.iloc[1:,5].fillna(x.mean()),axis=0)
#data.iloc[1:,5].fillna(data.iloc[1:,5].mean(),inplace=True)
##print(data.iloc[1][5])
#for i in range(1,len(data)):
 #  t = ((float(data.iloc[i][6]) - data.iloc[i][5])/abs(data.iloc[i][5]))*100.00
 #  if(t>0):
  #     print('1')
  # else:
   #    print('0')
   



#data['Fund2008'] = (data['Fund2008'].str.split()).apply(lambda x: float(x[0].replace(',', '')))
#data.infer_objects().dtypes




#pd.to_numeric(data['Fund2008'].values.tolist(),errors='ignore')






def handle_non_numerical_data(data):
    columns = data.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if  data[column].dtype != np.int64 and data[column].dtype != np.float64:
            column_contents = data[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            data[column] = list(map(convert_to_int, data[column]))

    return data

data = handle_non_numerical_data(data)



X = data[['Fund2008','Fund2009','Fund2010']]
y= data[['Unnamed: 8']]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#print(data.iloc[1:,5].dtypes)

#print(data.head())
#data.iloc[1:,5].plot()

#plt.show()
