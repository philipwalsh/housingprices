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
from sklearn.ensemble import RandomForestClassifier


bVerbose = True
bSaveIntermediateWork = True
bLoadFullData = False

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

if bLoadFullData:
    train_data = pd.read_csv('excluded/train_full.csv', low_memory=False)
else:
    train_data = pd.read_csv('excluded/train.csv', low_memory=False)

print('\ntrain data')
print(train_data.head(2))

sns.pairplot(train_data)


# for now, just deal with training data
if bLoadFullData:
    test_data = pd.read_csv('excluded/test_full.csv', low_memory=False)
else:
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


def CleanData(clean_me_df):
    #'Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA','RRNe'
    clean_me_df.loc[(clean_me_df['Condition1'] == 'Norm'), 'C1_Norm'] = 1
    clean_me_df.loc[(clean_me_df['Condition1'] == 'Feedr'), 'C1_Feedr'] = 1
    clean_me_df.loc[(clean_me_df['Condition1'] == 'PosN'), 'C1_PosN'] = 1
    clean_me_df.loc[(clean_me_df['Condition1'] == 'Artery'), 'C1_Artery'] = 1
    clean_me_df.loc[(clean_me_df['Condition1'] == 'RRAe'), 'C1_RRAe'] = 1
    clean_me_df.loc[(clean_me_df['Condition1'] == 'RRNn'), 'C1_RRNn'] = 1
    clean_me_df.loc[(clean_me_df['Condition1'] == 'RRAn'), 'C1_RRAn'] = 1
    clean_me_df.loc[(clean_me_df['Condition1'] == 'PosA'), 'C1_PosA'] = 1
    clean_me_df.loc[(clean_me_df['Condition1'] == 'RRNe'), 'C1_RRNe'] = 1
    clean_me_df['C1_Norm'].fillna(0, inplace=True)
    clean_me_df['C1_Feedr'].fillna(0, inplace=True)
    clean_me_df['C1_PosN'].fillna(0, inplace=True)
    clean_me_df['C1_Artery'].fillna(0, inplace=True)
    clean_me_df['C1_RRAe'].fillna(0, inplace=True)
    clean_me_df['C1_RRNn'].fillna(0, inplace=True)
    clean_me_df['C1_RRAn'].fillna(0, inplace=True)
    clean_me_df['C1_PosA'].fillna(0, inplace=True)
    clean_me_df['C1_RRNe'].fillna(0, inplace=True)
    
    clean_me_df['Electrical'].fillna('SBrkr', inplace=True)
    
    clean_me_df.loc[(clean_me_df['Electrical'] == 'SBrkr'), 'E_SBrkr'] = 1
    clean_me_df.loc[(clean_me_df['Electrical'] == 'FuseF'), 'E_FuseF'] = 1
    clean_me_df.loc[(clean_me_df['Electrical'] == 'FuseA'), 'E_FuseA'] = 1
    clean_me_df.loc[(clean_me_df['Electrical'] == 'FuseP'), 'E_FuseP'] = 1
    clean_me_df.loc[(clean_me_df['Electrical'] == 'Mix'), 'E_Mix'] = 1
    clean_me_df['E_SBrkr'].fillna(0, inplace=True)
    clean_me_df['E_FuseF'].fillna(0, inplace=True)
    clean_me_df['E_FuseA'].fillna(0, inplace=True)
    clean_me_df['E_FuseP'].fillna(0, inplace=True)
    clean_me_df['E_Mix'].fillna(0, inplace=True)
    
    
    clean_me_df.loc[(clean_me_df['GarageCars'] == 1), 'GC_1'] = 1
    clean_me_df.loc[(clean_me_df['GarageCars'] == 2), 'GC_2'] = 1
    clean_me_df.loc[(clean_me_df['GarageCars'] == 3), 'GC_3'] = 1
    clean_me_df.loc[(clean_me_df['GarageCars'] == 4), 'GC_4'] = 1
    clean_me_df.loc[(clean_me_df['GarageCars'] == 5), 'GC_5'] = 1
    clean_me_df['GC_1'].fillna(0, inplace=True)
    clean_me_df['GC_2'].fillna(0, inplace=True)
    clean_me_df['GC_3'].fillna(0, inplace=True)
    clean_me_df['GC_4'].fillna(0, inplace=True)
    clean_me_df['GC_5'].fillna(0, inplace=True)
    
    clean_me_df['MSZoning'].fillna('RL', inplace=True) #'RM' 'C (all)' 'FV' 'RH'
    clean_me_df.loc[(clean_me_df['MSZoning'] == 'RL'), 'MSZ_RL'] = 1
    clean_me_df.loc[(clean_me_df['MSZoning'] == 'RM'), 'MSZ_RM'] = 1
    clean_me_df.loc[(clean_me_df['MSZoning'] == 'C (all)'), 'MSZ_C'] = 1
    clean_me_df.loc[(clean_me_df['MSZoning'] == 'FV'), 'MSZ_FV'] = 1
    clean_me_df.loc[(clean_me_df['MSZoning'] == 'RH'), 'MSZ_RH'] = 1
    clean_me_df['MSZ_RL'].fillna(0, inplace=True)
    clean_me_df['MSZ_RM'].fillna(0, inplace=True)
    clean_me_df['MSZ_C'].fillna(0, inplace=True)
    clean_me_df['MSZ_FV'].fillna(0, inplace=True)
    clean_me_df['MSZ_RH'].fillna(0, inplace=True)
    
    print('cleaned!')


CleanData(train_data)
if bSaveIntermediateWork:
    print('saving cleaned data ...', sendtofile(excluded_dir,'cleaned_df.csv',train_data))


print(train_data.columns)
print(1/1)
#lets have a peak at whats important

#train_cols=['LotArea','C1_Norm', 'C1_Feedr','C1_PosN', 'C1_Artery', 'C1_RRAe','C1_RRNn','C1_RRAn', 'C1_PosA','C1_RRNe','E_SBrkr','E_FuseF','E_FuseA','E_FuseP','E_Mix','GC_1','GC_2','GC_3','GC_4','GC_5','MSZ_RL','MSZ_RM','MSZ_C','MSZ_FV','MSZ_RH']
train_cols=['LotArea','LotFrontage']

x_train = train_data[complete_df['SalePrice'] >= 0][train_cols]
y_train = train_data[complete_df['SalePrice'] >= 0]['SalePrice']

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(x_train, y_train.values.ravel())
features = pd.DataFrame()
features['feature'] = x_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
print(features)