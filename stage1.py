import pandas as pd


file = r'data.xls'
data = pd.read_excel(file)


data.fillna(0,inplace=True)
data.iloc[1:,5].replace(0,1,inplace=True)
#data.apply(lambda x: data.iloc[1:,5].fillna(x.mean()),axis=0)
#data.iloc[1:,5].fillna(data.iloc[1:,5].mean(),inplace=True)
##print(data.iloc[1][5])
for i in range(1,len(data)):
   t = ((float(data.iloc[i][6]) - data.iloc[i][5])/abs(data.iloc[i][5]))*100.00
   if(t>0):
       print('1')
   else:
       print('0')
    
