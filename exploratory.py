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
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



bVerbose = True
bSaveIntermediateWork = True
bLoadFullData = True

def sendtofile(outdir, filename, df):
    script_name='exploratory_'
    out_file = os.path.join(outdir, script_name + filename) 
    df.to_csv(out_file, index=False, header=True)
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

# we know the sale price is skewed, immediate deal with that
# train_data["SalePrice"] = np.log1p(train_data["SalePrice"]) 


print('\ntrain data')
print(train_data.head(2))

#sns.pairplot(train_data)


# for now, just deal with training data
if bLoadFullData:
    test_data = pd.read_csv('excluded/test_full.csv', low_memory=False)
else:
    test_data = pd.read_csv('excluded/test.csv', low_memory=False)



print('\ntest data')
print(test_data.head(10))

# combine the data sets into one so we can do the 
# cleanup and feature engineering only once
print('combining test and train...')
#complete_df = pd.concat([train_data,test_data], axis='rows', sort=False)

all_data = pd.concat((train_data.loc[:,'MSSubClass':'SaleCondition'],
                      test_data.loc[:,'MSSubClass':'SaleCondition']))



if bSaveIntermediateWork:
    print('saving combined data ...', sendtofile(excluded_dir,'all_data.csv',all_data))


def CleanData(clean_me_df):


    temp_mean = clean_me_df["LotFrontage"].mean()
    clean_me_df['LotFrontage'].fillna(temp_mean, inplace=True)

    temp_mean = clean_me_df["GrLivArea"].mean()
    clean_me_df['GrLivArea'].fillna(temp_mean, inplace=True)

    numeric_feats = clean_me_df.dtypes[all_data.dtypes != "object"].index
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


CleanData(all_data)


if bSaveIntermediateWork:
    print('saving cleaned data ...', sendtofile(excluded_dir,'all_data(cleaned).csv',all_data))



print("checkpoint", 1/1)
#lets have a peak at whats important

#train_cols=['LotArea','C1_Norm', 'C1_Feedr','C1_PosN', 'C1_Artery', 'C1_RRAe','C1_RRNn','C1_RRAn', 'C1_PosA','C1_RRNe','E_SBrkr','E_FuseF','E_FuseA','E_FuseP','E_Mix','GC_1','GC_2','GC_3','GC_4','GC_5','MSZ_RL','MSZ_RM','MSZ_C','MSZ_FV','MSZ_RH']
train_cols=['LotArea','LotFrontage','GrLivArea','C1_Norm', 'C1_Feedr','C1_PosN', 'C1_Artery', 'C1_RRAe','C1_RRNn','C1_RRAn', 'C1_PosA','C1_RRNe','E_SBrkr','E_FuseF','E_FuseA','E_FuseP','E_Mix','GC_1','GC_2','GC_3','GC_4','GC_5','MSZ_RL','MSZ_RM','MSZ_C','MSZ_FV','MSZ_RH']


X_train = all_data[:train_data.shape[0]]
X_test = all_data[train_data.shape[0]:]
y = train_data.SalePrice

if bSaveIntermediateWork:
    print('saving X_train[train_cols]...', sendtofile(excluded_dir,'X_train(train_cols).csv',X_train[train_cols]))
    print('saving y...', sendtofile(excluded_dir,'y.csv',y))


print("X_train.head()")
print(X_train.head())
print("x_test.head()")
print(X_test.head())
print("X_train[train_cols].head()")
print(X_train[train_cols].head())

print("checkpoint", 1/1)


#clf = RandomForestRegressor(n_estimators=50, max_features='sqrt')
#clf = clf.fit(X_train[train_cols], y.values.ravel())
#features = pd.DataFrame()
#features['feature'] = X_train.columns
#features['importance'] = clf.feature_importances_
#features.sort_values(by=['importance'], ascending=True, inplace=True)
#features.set_index('feature', inplace=True)
#print(features)
# https://www.kaggle.com/apapiu/regularized-linear-models
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train[train_cols], y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

if False:
    model_ridge = Ridge()
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
                for alpha in alphas]
    cv_ridge = pd.Series(cv_ridge, index = alphas)
    cv_ridge.plot(title = "Validation - Just Do It")
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    plt.show()
    print("cv_ridge.min()", cv_ridge.min())

# from the chart we see the best alpha is about 10

# now we try lasso, it will find best alpha and best features for us
if False:
    model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train[train_cols], y)
    print("rmse_cv(model_lasso).mean()", rmse_cv(model_lasso).mean())

if False:
    #lets have a look at what the most important vars are
    coef = pd.Series(model_lasso.coef_, index = X_train[train_cols].columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
    #plt.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Coefficients in the Lasso Model")
    plt.show()


if False:
    preds = pd.DataFrame({"preds":model_lasso.predict(X_train[train_cols]), "true":y})
    preds["residuals"] = preds["true"] - preds["preds"]
    preds.plot(x = "preds", y = "residuals",kind = "scatter")
    plt.show()

if False:
    submission_y = test_data.Id
    pred_y=model_lasso.predict(X_test[train_cols])
    pred_df=pd.DataFrame(pred_y, columns=['SalePrice'])
    print(pred_df.head())
    submission_df = pd.concat([submission_y,pred_df], axis='columns', sort=False)
    print('saving submission_df ...', sendtofile(excluded_dir,'submission_df.csv',submission_df))




# X_train = all_data[:train_data.shape[0]]
# X_test = all_data[train_data.shape[0]:]
# y = train_data.SalePrice

