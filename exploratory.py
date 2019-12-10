# phil walsh
# 2019-12-10
# https://twitter.com/PhilipWalsh_ML
# https://www.kaggle.com/pmcphilwalsh
# https://pmcphilwalsh.github.io/
# http://www.philwalsh.com

import numpy as np
import pandas as pd
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import seaborn as sns


bVerbose = True
bSaveIntermediateWork = True
def sendtofile(outdir, filename, df):
    script_name='exploratory_'
    out_file = os.path.join(excluded_dir, script_name + filename) 
    df.to_csv(out_file, index=False)
    return out_file


#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', -1)

working_dir=os.getcwd()
excluded_dir = os.path.join(working_dir, 'excluded') # working_dir + '\excluded'
if bVerbose:
    print('\nWorking Dir    :', working_dir)
    print('Excluded Dir   :', excluded_dir)

onlyfiles = [f for f in os.listdir(excluded_dir) if isfile(join(excluded_dir, f))]

print('\nFiles')
print(onlyfiles)

print('loading train data...')

train_data = pd.read_csv('excluded/train.csv', low_memory=False)

print('\ntrain data')
print(train_data.head(2))

sns.pairplot(train_data)


# for now, just deal with training data
if False:
    test_data = pd.read_csv('excluded/test.csv', low_memory=False)
    test_data['SalePrice']=-1
    print('\ntest data')
    print(test_data.head(10))

    # combine the data sets into one so we can do the 
    # cleanup and feature engineering only once
    print('combining test and train...')

    complete_df = pd.concat([train_data,test_data], axis='rows', sort=False)

    if bSaveIntermediateWork:
        print('saving combined data ...', sendtofile(excluded_dir,'complete_df.csv',complete_df))


