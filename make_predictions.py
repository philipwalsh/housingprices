import numpy as np
import pandas as pd
import os
from os.path import isfile, join
from scipy.stats import skew
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression



working_dir=os.getcwd()
excluded_dir = os.path.join(working_dir, 'excluded') # working_dir + '\excluded'


def sendtofile(outdir, filename, df):
    script_name="hp_preds_logreg_"
    out_file = os.path.join(outdir, script_name + filename) 
    df.to_csv(out_file, index=False)
    return out_file


def CleanData(clean_me_df):


    temp_mean = clean_me_df["LotFrontage"].mean()
    clean_me_df['LotFrontage'].fillna(temp_mean, inplace=True)

    temp_mean = clean_me_df["GrLivArea"].mean()
    clean_me_df['GrLivArea'].fillna(temp_mean, inplace=True)

    numeric_feats = clean_me_df.dtypes[clean_me_df.dtypes != "object"].index
    skewed_feats = clean_me_df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats =skewed_feats>0.75
    skewed_index = skewed_feats.index
    clean_me_df[skewed_index] = np.log1p(clean_me_df[skewed_index])

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
    
    #blast out any remaining misisng data
    clean_me_df = clean_me_df.fillna(clean_me_df.mean())
    print('cleaned!')
    return clean_me_df


train_data = pd.read_csv("excluded/train_full.csv", low_memory=False)
test_data = pd.read_csv("excluded/test_full.csv", low_memory=False)


train_y = train_data[['SalePrice']]
clean_df = CleanData(train_data)

fit_cols=['GrLivArea','C1_RRNn','MSZ_FV','E_SBrkr','C1_Norm','MSZ_RL','LotArea','LotFrontage','C1_RRAn','E_FuseA']
train_x = clean_df[fit_cols]

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.25)   

model = LinearRegression()
print("x_train.shape   :", x_train.shape)
print("y_train.shape   :", y_train.shape)
y_reshaped=y_train.values.ravel()
print("y_reshaped.shape   :", y_reshaped.shape)

model.fit(x_train, y_reshaped)
myscore=model.score(x_test, y_test)
print("\nmyscore",myscore)


submission_y = test_data[['Id']]
test_data.drop("Id", axis="columns", inplace=True)
clean_submission_df = CleanData(test_data)

submission_x = clean_submission_df[fit_cols]



pred_y=model.predict(submission_x)
pred_df=pd.DataFrame(pred_y, columns=['SalePrice'])
submission_df = pd.concat([submission_y,pred_df], axis="columns", sort=False)
print("saving submission_df ...", sendtofile(excluded_dir,"lr_submission_df.csv",submission_df))
