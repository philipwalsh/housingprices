# phil walsh
# pmcphilwalsh@gmail.com
# this script is based off of make_predictions_2.py

# 2019-12-16
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from os.path import isfile, join
from scipy.stats import skew
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV



bVerbose = False




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

def CleanData(clean_me_df):


    clean_me_df.drop('Id', axis=1, inplace=True)
    LotFrontage_mean = 0
    LotFrontage_mean = clean_me_df['LotFrontage'].mean()
    clean_me_df['LotFrontage'].fillna(LotFrontage_mean, inplace=True)

    GrvLivArea_mean = 0
    GrvLivArea_mean = clean_me_df['GrLivArea'].mean()
    clean_me_df['GrLivArea'].fillna(GrvLivArea_mean, inplace=True)
    
    

    YearBullt_mean=0
    YearBullt_mean = clean_me_df['YearBuilt'].mean()
    clean_me_df['YearBuilt'].fillna(YearBullt_mean, inplace=True)    
    clean_me_df['YearDecade'] = clean_me_df['YearBuilt'].astype(int).map(lambda x: get_decade(x))
    clean_me_df['DecadesOld'] = 200-(2010 - clean_me_df['YearDecade'])
    clean_me_df.drop('YearBuilt', axis=1, inplace=True)
    clean_me_df.drop('YearDecade', axis=1, inplace=True)    

    YearRemodAdd_mean=0
    YearRemodAdd_mean = clean_me_df['YearRemodAdd'].mean()
    clean_me_df['YearRemodAdd'].fillna(YearRemodAdd_mean, inplace=True)    
    clean_me_df['YearRemodAddDecade'] = clean_me_df['YearRemodAdd'].astype(int).map(lambda x: get_decade(x))
    clean_me_df['YearRemodAddDecOld'] = 200-(2010 - clean_me_df['YearRemodAddDecade'])
    clean_me_df.drop('YearRemodAdd', axis=1, inplace=True)
    clean_me_df.drop('YearRemodAddDecade', axis=1, inplace=True)    
    
    GarageYrBlt_mean=0
    GarageYrBlt_mean = clean_me_df['GarageYrBlt'].mean()
    clean_me_df['GarageYrBlt'].fillna(GarageYrBlt_mean, inplace=True)    
    clean_me_df['GarageYrBltDecade'] = clean_me_df['GarageYrBlt'].astype(int).map(lambda x: get_decade(x))
    clean_me_df['GarageYrBltDecOld'] = 200-(2010 - clean_me_df['GarageYrBltDecade'])
    clean_me_df.drop('GarageYrBlt', axis=1, inplace=True)
    clean_me_df.drop('GarageYrBltDecade', axis=1, inplace=True)    
    

    
    
    clean_me_df['Condition1'].fillna('Norm', inplace=True)
    clean_me_df['Condition2'].fillna('Norm', inplace=True)
    clean_me_df['Neighborhood'].fillna('NAmes', inplace=True) 
    clean_me_df['GarageQual'].fillna('TA', inplace=True)    
    clean_me_df['GarageCond'].fillna('TA', inplace=True)
    clean_me_df['LotShape'].fillna('Reg', inplace=True)    
    clean_me_df['LandContour'].fillna('Lvl', inplace=True) 
    clean_me_df['LotConfig'].fillna('Inside', inplace=True)
    clean_me_df['LandSlope'].fillna('Gtl', inplace=True)
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

    # engineered features
    # total bathroom count
    # BsmtFullBath + BsmtHalfBath * .5 + FullBath + HalfBath
    # pricer per bathroom
    # cleanme['GLVPerBath']=clean_me_df['GrLivArea']/BathroomCount
    #YearRemodAdd
    #GarageYrBlt

    



    return clean_me_df



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
combined = CleanData(combined)
sendtofile(excluded_dir, 'combined.csv', combined, verbose=True)

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
train_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
'3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 
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
'SaleCondition_Normal', 'SaleCondition_Partial']#,'DecadesOld','YearRemodAddDecOld','GarageYrBltDecOld']


# I should probably cull down the vars, and do some feature engineering, WIP


#####
##### FIT THE MODELS
#####




sendtofile(excluded_dir,"X_train.csv",X_train[train_cols], verbose=True)
sendtofile(excluded_dir,"y_train.csv",pd.DataFrame(y_train), verbose=True)
sendtofile(excluded_dir,"X_test.csv",X_test[train_cols], verbose=True)
sendtofile(excluded_dir,"y_test.csv",pd.DataFrame(y_test), verbose=True)


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


#set to True if you want to play with optimization
# set to False when ready to run
if False:
    param_grid={
        'n_estimators':[301, 279], 
        'learning_rate': [0.05, 0.025],# , 0.01],
        'max_depth':[3, 4, 5], 
        'min_samples_leaf':[5],#9,17], 
        'max_features':[0.25] 
        } 
    n_jobs=-1 


    #print("Feature Importances")
    #print (model_gb.feature_importances_) 
    estimator = GradientBoostingRegressor(random_state=9261774) 
    # cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2) 
    classifier = GridSearchCV(estimator=estimator,cv=5, param_grid=param_grid, n_jobs=n_jobs) 
    #Also note that we're feeding multiple neighbors to the GridSearch to try out. 
    #We'll now fit the training dataset to this classifier 
    classifier.fit(X_train[train_cols], y_train) 
    #Let's look at the best estimator that was found by GridSearchCV print "Best Estimator learned through GridSearch" 
    print (classifier.best_estimator_ )
    print (classifier.best_params_ )
    print (classifier.best_score_ )
else:
    # {'learning_rate': 0.05, 'max_depth': 4, 'max_features': 1.0, 'min_samples_leaf': 5, 'n_estimators': 221}
    # 0.855131319451837
    # {'learning_rate': 0.05, 'max_depth': 4, 'max_features': 0.25, 'min_samples_leaf': 5, 'n_estimators': 279}
    # 0.86858088329903
    # {'learning_rate': 0.025, 'max_depth': 5, 'max_features': 0.25, 'min_samples_leaf': 5, 'n_estimators': 301}
    # 0.8718522939695451

    model_gb = GradientBoostingRegressor(random_state=9261774,n_estimators=100)#,learning_rate=0.05, max_depth=4, max_features=0.25, min_samples_leaf= 5, n_estimators=279)
    model_gb.fit(X_train[train_cols], y_train)
    train_score_gb=model_gb.score(X_train[train_cols], y_train)
    test_score_gb=model_gb.score(X_test[train_cols], y_test)
    print('gb training score     : ', train_score_gb)
    print('gb test score         : ', test_score_gb)

    #before optimization
    #gb training score     :  0.9648429174615337
    #gb test score         :  0.8447637548701106

    #after optimization
    #gb training score     :  0.9802699819779025
    #gb test score         :  0.8524618815557955



#####
##### Create Submission
#####
if True:

    #Linear Regression
    pred_id = sub_data[['Id']]
    # get predictions
    pred_y_lr=model_lr.predict(X_sub[train_cols])
    # tack the saved labes (y's) onto the preds into a data frame
    pred_lr=pd.DataFrame(pred_y_lr, columns=['SalePrice'])
    submission_lr = pd.concat([submission_id,pred_lr], axis='columns', sort=False)
    sendtofile(excluded_dir,'predictions_lr.csv',submission_lr, verbose=True)

    #random forest
    pred_y_rf=model_rf.predict(X_sub[train_cols])
    #tack the saved labes (y's) onto the preds into a data frame
    pred_rf=pd.DataFrame(pred_y_rf, columns=['SalePrice'])
    submission_rf = pd.concat([submission_id,pred_rf], axis='columns', sort=False)
    sendtofile(excluded_dir,'predictions_rf.csv',submission_rf, verbose=True)



    #GradientBoost
    
    pred_y_gb=model_gb.predict(X_sub[train_cols])
    #tack the saved labes (y's) onto the preds into a data frame
    pred_gb=pd.DataFrame(pred_y_gb, columns=['SalePrice'])
    submission_gb = pd.concat([submission_id,pred_gb], axis='columns', sort=False)
    sendtofile(excluded_dir,'predictions_gb.csv',submission_gb, verbose=True)



    
    submission_lr_rf = pd.concat([submission_lr,pred_rf], axis='columns', sort=False)
    submission_lr_rf_gb = pd.concat([submission_lr_rf,pred_gb], axis='columns', sort=False)
    submission_lr_rf_gb.columns=['Id','SalePrice_LR', 'SalePrice_RF','SalePrice_GB']
    submission_lr_rf_gb['SalePrice']=(submission_lr_rf_gb['SalePrice_LR']+submission_lr_rf_gb['SalePrice_RF']+submission_lr_rf_gb['SalePrice_GB'])/3
    submission_lr_rf_gb.drop('SalePrice_LR', axis=1, inplace=True)
    submission_lr_rf_gb.drop('SalePrice_RF', axis=1, inplace=True)
    submission_lr_rf_gb.drop('SalePrice_GB', axis=1, inplace=True)
    sendtofile(excluded_dir,'predictions_lr_rf_gb.csv',submission_lr_rf_gb, verbose=True)


print('\n\n*****')
print('***** end of sript: make_preds_3.py')
print('*****')

# 2019-12-16 after removing year built, remodel year and garage year built
# lm training score     :  0.9432573451774633
# lm test score         :  0.8136472479742833
# rf training score     :  0.981845737882965
# rf test score         :  0.8338866779653443
# kaggel score          : 0.14019  * best yet


# 2019-12-16 after removing year built, remodel year and garage year built
# lm training score     :  0.9432573451774633
# lm test score         :  0.8136472479742833
# rf training score     :  0.981845737882965
# rf test score         :  0.8338866779653443
# gb training score     :  0.9648429174615337
# gb test score         :  0.8447637548701106
# kaggel score          :  0.13544  * best yet, this is (lr+rf+gb)/3 ensemble
# i wonder if the lm or rf is holding me back, i will try ensemble of gb and lm, then gb and rf
# lr and gb             :  0.14306
# rf and gb             :  0.14183

