def get_decade (value):
    return int(value / 10) * 10

print(get_decade(1968))


import numpy as np
import pandas as pd
import os

working_dir=os.getcwd()
excluded_dir = os.path.join(working_dir, 'excluded') # working_dir + '\excluded'


train_data = pd.read_csv('excluded/train.csv', low_memory=False)
print(train_data.columns[9])
