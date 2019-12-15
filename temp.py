def get_decade (value):
    return int(value / 10) * 10

print(get_decade(1968))


import numpy as np
import pandas as pd
import os

working_dir=os.getcwd()
excluded_dir = os.path.join(working_dir, 'excluded') # working_dir + '\excluded'


df = pd.read_csv('excluded/train.csv', low_memory=False)
#print(train_data.columns[9])
x = df.select_dtypes(include=np.object).columns.tolist()
for n in x:
    df1 = pd.get_dummies(df[n])
    df = pd.concat([df,df1], axis=1)

df.to_csv("excluded\\df_dummies_test.csv", index=False)

