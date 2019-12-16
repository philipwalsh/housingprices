# phil walsh
# pmcphilwalsh@gmail.com
# this script is based off of make_predictions_2.py
# goals for this script
#  1) linear reg and random forest, ensembled
#  2) one hot encode all the non numeric!

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from os.path import isfile, join
from scipy.stats import skew
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression



bVerbose = False

LotFrontage_mean = -1
GrvLivArea_mean = -1
YearBullt_mean = -1



working_dir=os.getcwd()
excluded_dir = os.path.join(working_dir, 'excluded') # working_dir + '\excluded'


print('\n\n*****')
print('***** start of sript: make_preds_3.py')
print('*****')

if bVerbose:
    print('\nworking dir   :', working_dir)



#print('saving X_sub ...', sendtofile(excluded_dir,'submisison_data(cleaned).csv',X_sub))
def sendtofile(outdir, filename, df, verbose=False):
    script_name='eval_models_5_'
    out_file = os.path.join(outdir, script_name + filename)
    if verbose:
        print("saving file :", out_file)
    df.to_csv(out_file, index=False)
    return out_file


def get_decade(in_val):
    return int(in_val / 10) * 10

def CleanData(clean_me_df,LotFrontage_mean,GrvLivArea_mean,YearBullt_mean):


    clean_me_df.drop('Id', axis=1, inplace=True)
    if LotFrontage_mean < 0:
        LotFrontage_mean = clean_me_df['LotFrontage'].mean()
    clean_me_df['LotFrontage'].fillna(LotFrontage_mean, inplace=True)

    if GrvLivArea_mean < 0:
        GrvLivArea_mean = clean_me_df['GrLivArea'].mean()
    clean_me_df['GrLivArea'].fillna(GrvLivArea_mean, inplace=True)
    

    if YearBullt_mean<0:
        YearBullt_mean = clean_me_df['YearBuilt'].mean()
    clean_me_df['YearBuilt'].fillna(YearBullt_mean, inplace=True)    
    clean_me_df['YearDecade'] = clean_me_df['YearBuilt'].astype(int).map(lambda x: get_decade(x))
    clean_me_df['DecadesOld'] = 2010 - clean_me_df['YearDecade']
    clean_me_df.drop('YearBuilt', axis=1, inplace=True)
    clean_me_df.drop('DecadesOld', axis=1, inplace=True)    
    
    
    clean_me_df['Condition1'].fillna('Norm', inplace=True)
    clean_me_df['Condition2'].fillna('Norm', inplace=True)
    clean_me_df['Neighborhood'].fillna('NAmes', inplace=True) 
    clean_me_df['GarageQual'].fillna('TA', inplace=True)    
    clean_me_df['GarageCond'].fillna('TA', inplace=True)
    clean_me_df['LotShape'].fillna('Reg', inplace=True)    
    clean_me_df['LandContour'].fillna('Lvl', inplace=True) 
    clean_me_df['LotConfig'].fillna('Inside', inplace=True)
    clean_me_df['LandSlope'].fillna('Gtl', inplace=True)
    clean_me_df['GarageYrBlt'].fillna(0, inplace=True)
    clean_me_df['GarageCars'].fillna(0, inplace=True)
    clean_me_df['GarageArea'].fillna(0, inplace=True)
    clean_me_df['BsmtHalfBath'].fillna(0, inplace=True)
    clean_me_df['BsmtFullBath'].fillna(0, inplace=True)
    clean_me_df['BsmtFinSF1'].fillna(0, inplace=True)
    clean_me_df['BsmtFinSF2'].fillna(0, inplace=True)
    clean_me_df['BsmtUnfSF'].fillna(0, inplace=True)
    clean_me_df['TotalBsmtSF'].fillna(0, inplace=True)
    clean_me_df['MSZoning'].fillna('RL', inplace=True)
    clean_me_df['MasVnrArea'].fillna(0, inplace=True)

    #one hot encoding for all object types
    x = clean_me_df.select_dtypes(include=np.object).columns.tolist()
    for n in x:
        df1 = pd.get_dummies(clean_me_df[n], prefix = n)
        clean_me_df = pd.concat([clean_me_df,df1], axis=1)

    # drop the columns that have been encoded
    for n in x:
        clean_me_df.drop(n,axis=1, inplace=True)

    
    return clean_me_df,LotFrontage_mean,GrvLivArea_mean,YearBullt_mean



train_data = pd.read_csv('excluded/train_full.csv', low_memory=False)
sub_data = pd.read_csv('excluded/test_full.csv', low_memory=False)





# stratified shuffle split
# basically stratirfy the data by Gross Living Area
# get the test and hold-out data to jive with each other, regarding sampling based on these bins
# probably most useful in the linear model, and why it gives me a 94% training score and a 81% on the holdout
train_data['living_area_cat'] = pd.cut(
    train_data['GrLivArea'], 
    bins=[0, 1000, 1500, 2000, 2500, np.inf], 
    labels=[1, 2, 3, 4, 5])



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=9261774)
for train_index, test_index in split.split(train_data, train_data['living_area_cat']):
    X_train = train_data.loc[train_index] # this is the training data
    X_test = train_data.loc[test_index]   # this is the hold out, the protion of the training i will use for testing



y_train = X_train['SalePrice']
y_test = X_test['SalePrice']

X_train.drop('SalePrice', axis=1, inplace=True)
X_test.drop('SalePrice', axis=1, inplace=True)


submission_id = sub_data['Id']  # this is the start of the submission data frame.  
                                # sub data is already loaded, store the Id now, later we add in the y predictions


# drop the straify category, not needed anymore 
for set_ in (X_train, X_test, train_data):
    set_.drop('living_area_cat', axis=1, inplace=True)


# combine all of the data into one big fat hairy beast.  use this for cleaning the data
# so we dont have any surprises regarding one hot encoding
X_train['TRAIN']=1          # 1 indicates its from the training data
X_test['TRAIN']=0           # 0 indicates its hold-out
sub_data['TRAIN']=-1        # -1 for the submissions data


# combine it all together
combined=pd.concat([X_train, X_test, sub_data])

#####
#####  Clean The Data
#####

# this will do the heavy lifting removing NaN(s), dropping columns, one hot encoding and feature engineering
combined, LotFrontage_mean, GrvLivArea_mean, YearBullt_mean = CleanData(combined,LotFrontage_mean,GrvLivArea_mean,YearBullt_mean)

#####
##### Put the data apart into the proper data frames. Train, Test (aka holdout) and Submission
#####

# train - use to fit models
X_train = combined[combined['TRAIN']==1].copy()
X_train.drop('TRAIN', axis=1, inplace=True)

# test - use to evaluate model performance
X_test = combined[combined['TRAIN']==0].copy()
X_test.drop('TRAIN', axis=1, inplace=True)

# sub - use to submit the final answer for the kaggle cometition
X_sub = combined[combined['TRAIN']==-1].copy()
X_sub.drop('TRAIN', axis=1, inplace=True)

#all cols
train_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
'3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'YearDecade', 'MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 
'MSZoning_RM', 'Street_Grvl', 'Street_Pave', 'Alley_Grvl', 'Alley_Pave', 'LotShape_IR1', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 
'LandContour_Bnk', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'Utilities_AllPub', 'Utilities_NoSeWa', 'LotConfig_Corner', 'LotConfig_CulDSac', 
'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside', 'LandSlope_Gtl', 'LandSlope_Mod', 'LandSlope_Sev', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 
'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 
'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill', 
'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 
'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'Condition1_Artery', 
'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 
'Condition2_Artery', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_PosA', 'Condition2_PosN', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 
'BldgType_1Fam', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 
'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'RoofStyle_Flat', 'RoofStyle_Gable', 'RoofStyle_Gambrel', 
'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'RoofMatl_ClyTile', 'RoofMatl_CompShg', 'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll', 
'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'Exterior1st_AsbShng', 'Exterior1st_AsphShn', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 
'Exterior1st_CBlock', 'Exterior1st_CemntBd', 'Exterior1st_HdBoard', 'Exterior1st_ImStucc', 'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_Stone', 
'Exterior1st_Stucco', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior1st_WdShing', 'Exterior2nd_AsbShng', 'Exterior2nd_AsphShn', 'Exterior2nd_Brk Cmn', 
'Exterior2nd_BrkFace', 'Exterior2nd_CBlock', 'Exterior2nd_CmentBd', 'Exterior2nd_HdBoard', 'Exterior2nd_ImStucc', 'Exterior2nd_MetalSd', 'Exterior2nd_Other', 
'Exterior2nd_Plywood', 'Exterior2nd_Stone', 'Exterior2nd_Stucco', 'Exterior2nd_VinylSd', 'Exterior2nd_Wd Sdng', 'Exterior2nd_Wd Shng', 'MasVnrType_BrkCmn', 
'MasVnrType_BrkFace', 'MasVnrType_None', 'MasVnrType_Stone', 'ExterQual_Ex', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'ExterCond_Ex', 'ExterCond_Fa', 
'ExterCond_Gd', 'ExterCond_Po', 'ExterCond_TA', 'Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 
'Foundation_Wood', 'BsmtQual_Ex', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA', 'BsmtCond_Fa', 'BsmtCond_Gd', 'BsmtCond_Po', 'BsmtCond_TA', 'BsmtExposure_Av', 
'BsmtExposure_Gd', 'BsmtExposure_Mn', 'BsmtExposure_No', 'BsmtFinType1_ALQ', 'BsmtFinType1_BLQ', 'BsmtFinType1_GLQ', 'BsmtFinType1_LwQ', 'BsmtFinType1_Rec', 
'BsmtFinType1_Unf', 'BsmtFinType2_ALQ', 'BsmtFinType2_BLQ', 'BsmtFinType2_GLQ', 'BsmtFinType2_LwQ', 'BsmtFinType2_Rec', 'BsmtFinType2_Unf', 'Heating_Floor', 
'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'HeatingQC_Ex', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_Po', 'HeatingQC_TA', 
'CentralAir_N', 'CentralAir_Y', 'Electrical_FuseA', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'Electrical_SBrkr', 'KitchenQual_Ex', 'KitchenQual_Fa', 
'KitchenQual_Gd', 'KitchenQual_TA', 'Functional_Maj1', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 
'FireplaceQu_Ex', 'FireplaceQu_Fa', 'FireplaceQu_Gd', 'FireplaceQu_Po', 'FireplaceQu_TA', 'GarageType_2Types', 'GarageType_Attchd', 'GarageType_Basment', 
'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'GarageFinish_Fin', 'GarageFinish_RFn', 'GarageFinish_Unf', 'GarageQual_Ex', 'GarageQual_Fa', 
'GarageQual_Gd', 'GarageQual_Po', 'GarageQual_TA', 'GarageCond_Ex', 'GarageCond_Fa', 'GarageCond_Gd', 'GarageCond_Po', 'GarageCond_TA', 'PavedDrive_N', 
'PavedDrive_P', 'PavedDrive_Y', 'PoolQC_Ex', 'PoolQC_Fa', 'PoolQC_Gd', 'Fence_GdPrv', 'Fence_GdWo', 'Fence_MnPrv', 'Fence_MnWw', 'MiscFeature_Gar2', 
'MiscFeature_Othr', 'MiscFeature_Shed', 'MiscFeature_TenC', 'SaleType_COD', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 
'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'SaleCondition_Abnorml', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 
'SaleCondition_Normal', 'SaleCondition_Partial']


# I should probably cull down the vars, and do some feature engineering, WIP


#####
##### FIT THE MODELS
#####

# Linear Regression
model_lr = LinearRegression(normalize=False)
if False:
    print('model_lr Parameters currently in use:\n')
    print(model_lr.get_params())

model_lr.fit(X_train[train_cols], y_train)
train_score_lm=model_lr.score(X_train[train_cols], y_train)

# Random forrest
model_rf = RandomForestRegressor(random_state=9261774,max_features=87, n_estimators=171)
if False:
    print('model_rf Parameters currently in use:\n')
    print(model_rf.get_params())

model_rf.fit(X_train[train_cols], y_train)

train_score_rf=model_rf.score(X_train[train_cols], y_train)
    

test_score_lm=model_lr.score(X_test[train_cols], y_test)
test_score_rf=model_rf.score(X_test[train_cols], y_test)


print('lm training score     : ', train_score_lm)
print('lm test score         : ', test_score_lm)

print('rf training score     : ', train_score_rf)
print('rf test score         : ', test_score_rf)

#####
##### Create Submission
#####
if False:

    #Linear Regression
    pred_id = sub_data[['Id']]
    # get predictions
    pred_y_lr=model_lr.predict(X_sub[train_cols])
    # tack the saved labes (y's) onto the preds into a data frame
    pred_lr=pd.DataFrame(pred_y_lr, columns=['SalePrice'])
    submission_lr = pd.concat([submission_id,pred_lr], axis='columns', sort=False)
    print('saving submission_lr ...', sendtofile(excluded_dir,'predictions_lr.csv',submission_lr))

    #get predictions
    pred_y_rf=model_rf.predict(X_sub[train_cols])
    #tack the saved labes (y's) onto the preds into a data frame
    pred_rf=pd.DataFrame(pred_y_rf, columns=['SalePrice'])
    submission_rf = pd.concat([pred_id,pred_rf], axis='columns', sort=False)

    print('saving submission_rf ...', sendtofile(excluded_dir,'predictions_rf.csv',submission_rf))


print('\n\n*****')
print('***** end of sript: make_preds_3.py')
print('*****')


