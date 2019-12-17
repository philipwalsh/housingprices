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

    # get rid of the Id column
    clean_me_df.drop('Id', axis=1, inplace=True)


    # deal with NaNs
    if False:
            nan_data = clean_me_df.isnull().sum().sort_values(ascending=False)
            print(nan_data[nan_data > 0]/len(clean_me_df))

    clean_me_df["IsRegularLotShape"] = (clean_me_df.LotShape == "Reg") * 1
    clean_me_df["IsLandLevel"] = (clean_me_df.LandContour == "Lvl") * 1
    clean_me_df["IsLandSlopeGntl"] = (clean_me_df.LandSlope == "Gtl") * 1
    clean_me_df["IsElectricalSBrkr"] = (clean_me_df.Electrical == "SBrkr") * 1
    clean_me_df["IsGarageDetached"] = (clean_me_df.GarageType == "Detchd") * 1
    clean_me_df["IsPavedDrive"] = (clean_me_df.PavedDrive == "Y") * 1
    clean_me_df["HasShed"] = (clean_me_df.MiscFeature == "Shed") * 1


    clean_me_df.loc[clean_me_df.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1
    clean_me_df.loc[clean_me_df.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1
    clean_me_df.loc[clean_me_df.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1
    clean_me_df.loc[clean_me_df.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1
    clean_me_df.loc[clean_me_df.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1
    clean_me_df["Neighborhood_Good"].fillna(0, inplace=True)

    # these 4 are enourmously NaN
    #PoolQC          0.996574
    #MiscFeature     0.964029
    #Alley           0.932169
    #Fence           0.804385
    # drop the extrme NaN(s)
    x=['PoolQC','MiscFeature','Alley','Fence']
    for n in x:
        clean_me_df.drop(n,axis=1, inplace=True)

    #These are partially NaN  
    #FireplaceQu     0.486468
    clean_me_df['FireplaceQu'].fillna('NA', inplace=True)
    
    #LotFrontage     0.166495
    LotFrontage_mean = 0
    LotFrontage_mean = clean_me_df['LotFrontage'].mean()
    clean_me_df['LotFrontage'].fillna(LotFrontage_mean, inplace=True)

    #GarageCond      0.054471
    clean_me_df['GarageCond'].fillna('TA', inplace=True)
    #GarageYrBlt     0.054471
    temp_mean = clean_me_df['GarageYrBlt'].mean()
    clean_me_df['GarageYrBlt'].fillna(temp_mean, inplace=True)
    #GarageFinish    0.054471
    clean_me_df['GarageFinish'].fillna('NA', inplace=True)    
    #GarageQual      0.054471
    clean_me_df['GarageQual'].fillna('NA', inplace=True)    
    #GarageType      0.053786
    clean_me_df['GarageType'].fillna('NA', inplace=True)    
    #BsmtCond        0.028092
    clean_me_df['BsmtCond'].fillna('NA', inplace=True)    
    #BsmtExposure    0.028092
    clean_me_df['BsmtExposure'].fillna('NA', inplace=True)    
    #BsmtQual        0.027749
    clean_me_df['BsmtQual'].fillna('NA', inplace=True)    
    #BsmtFinType2    0.027407
    clean_me_df['BsmtFinType2'].fillna('NA', inplace=True)    
    #BsmtFinType1    0.027064
    clean_me_df['BsmtFinType1'].fillna('NA', inplace=True)    
    #MasVnrType      0.008222
    clean_me_df['MasVnrType'].fillna('None', inplace=True)    
    #MasVnrArea      0.007879
    clean_me_df['MasVnrArea'].fillna(0, inplace=True)
    #MSZoning        0.001370
    clean_me_df['MSZoning'].fillna('RL', inplace=True)
    #Utilities       0.000685
    clean_me_df['Utilities'].fillna('AllPub', inplace=True)
    #BsmtHalfBath    0.000685
    clean_me_df['BsmtHalfBath'].fillna(0, inplace=True)
    #BsmtFullBath    0.000685
    clean_me_df['BsmtFullBath'].fillna(0, inplace=True)
    #Functional      0.000685
    clean_me_df['Functional'].fillna('Mod', inplace=True)
    #Exterior1st     0.000343
    clean_me_df['Exterior1st'].fillna('Plywood', inplace=True)
    #TotalBsmtSF     0.000343
    clean_me_df['TotalBsmtSF'].fillna(0, inplace=True)
    #BsmtUnfSF       0.000343
    clean_me_df['BsmtUnfSF'].fillna(0, inplace=True)
    #BsmtFinSF2      0.000343
    clean_me_df['BsmtFinSF2'].fillna(0, inplace=True)
    #GarageArea      0.000343
    clean_me_df['GarageArea'].fillna(0, inplace=True)
    #KitchenQual     0.000343
    clean_me_df['KitchenQual'].fillna('TA', inplace=True)
    #GarageCars      0.000343
    clean_me_df['GarageCars'].fillna(0, inplace=True)
    #BsmtFinSF1      0.000343
    clean_me_df['BsmtFinSF1'].fillna(0, inplace=True)
    #Exterior2nd     0.000343
    clean_me_df['Exterior2nd'].fillna('Plywood', inplace=True)
    #SaleType        0.000343
    clean_me_df['SaleType'].fillna('Oth', inplace=True)
    #Electrical      0.000343    
    clean_me_df['Electrical'].fillna('FuseA', inplace=True)
    

    # this didnt show up earlier as having NaN data, but out of an abundance of cauthion, make sure
    GrvLivArea_mean = 0
    GrvLivArea_mean = clean_me_df['GrLivArea'].mean()
    clean_me_df['GrLivArea'].fillna(GrvLivArea_mean, inplace=True)
    
    

    #same, triple check we dont let a NaN through
    YearBullt_mean=0
    YearBullt_mean = clean_me_df['YearBuilt'].mean()
    clean_me_df['YearBuilt'].fillna(YearBullt_mean, inplace=True)    
    max_age = clean_me_df['YearBuilt'].max()
    clean_me_df['HouseAge'] = 1/((max_age+1)-clean_me_df['YearBuilt'])
    clean_me_df.drop('YearBuilt', axis=1, inplace=True)


    YearRemodAdd_mean=0
    YearRemodAdd_mean = clean_me_df['YearRemodAdd'].mean()
    clean_me_df['YearRemodAdd'].fillna(YearRemodAdd_mean, inplace=True)    
    max_age = clean_me_df['YearRemodAdd'].max()
    clean_me_df['RemodelAge'] = 1/((max_age+1)-clean_me_df['YearRemodAdd'])
    clean_me_df['NewRemodel'] = (clean_me_df['YearRemodAdd']==clean_me_df['YrSold']) * 1
    clean_me_df.drop('YearRemodAdd', axis=1, inplace=True)
    

    GarageYrBlt_mean=0
    GarageYrBlt_mean = clean_me_df['GarageYrBlt'].mean()
    clean_me_df['GarageYrBlt'].fillna(GarageYrBlt_mean, inplace=True)    
    max_age = clean_me_df['GarageYrBlt'].max()
    clean_me_df['GarageAge'] = 1/((max_age+1)-clean_me_df['GarageYrBlt'])    

    clean_me_df['NewGarage'] = (clean_me_df['GarageYrBlt']==clean_me_df['YrSold']) * 1
    clean_me_df.drop('GarageYrBlt', axis=1, inplace=True)    

    #one hot encode these, the rest can be categorized
    x=['MSSubClass','MSZoning','Street','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','GarageType','SaleType', 'SaleCondition']
    for n in x:
        df1 = pd.get_dummies(clean_me_df[n], prefix = n)
        clean_me_df = pd.concat([clean_me_df,df1], axis=1)

    # drop the columns that have been encoded
    for n in x:
        clean_me_df.drop(n,axis=1, inplace=True)


    #categorizing
    x = clean_me_df.select_dtypes(include=np.object).columns.tolist()
    for n in x:
        clean_me_df[n] = clean_me_df[n].astype('category')


    #dataTypeSeries = pd.DataFrame(clean_me_df.dtypes)
    #dataTypeSeries.columns=['DataType']
    #print('Data type of each column of Dataframe :')
    #print(dataTypeSeries[dataTypeSeries['DataType']=='category'])

    
    #ExterQual|ExterCond|BsmtQual|BsmtCond|HeatingQC|KitchenQual|FireplaceQu|GarageQual|GarageCond
    x = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
    for n in x:
        clean_me_df[n] = clean_me_df[n].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, NA=1, Po=0))


    #Utilities
    clean_me_df['Utilities'] = clean_me_df['Utilities'].replace(dict(AllPub=3,NoSewr=2, NoSeWa=1, ELO=0))
    # |Condition1
    clean_me_df['Condition1'] = clean_me_df['Condition1'].replace(dict(Artery=5, Feedr=4, Norm=3, RRNn=0, RRAn=0, PosN=3, PosA=2, RRNe=0, RRAe=0))
    # |Condition2
    clean_me_df['Condition2'] = clean_me_df['Condition2'].replace(dict(Artery=5, Feedr=4, Norm=3, RRNn=0, RRAn=0, PosN=3, PosA=2, RRNe=0, RRAe=0))
    # BsmtExposure
    clean_me_df['BsmtExposure'] = clean_me_df['BsmtExposure'].replace(dict(Gd=3, Av=2, Mn=1, No=0, NA=0))
    # BsmtFinType1
    clean_me_df['BsmtFinType1'] = clean_me_df['BsmtFinType1'].replace(dict(GLQ=5, ALQ=4, BLQ=3, Rec=2, LwQ=1, Unf=0, NA=0))
    # BsmtFinType2
    clean_me_df['BsmtFinType2'] = clean_me_df['BsmtFinType2'].replace(dict(GLQ=5, ALQ=4, BLQ=3, Rec=2, LwQ=1, Unf=0, NA=0))
    # CentralAir
    clean_me_df['CentralAir'] = clean_me_df['CentralAir'].replace(dict(Y=1, N=0))
    # Electrical
    clean_me_df['Electrical'] = clean_me_df['Electrical'].replace(dict(SBrkr=4,FuseA=3, FuseF=2, FuseP=1, Mix=0))
    # Functional
    clean_me_df['Functional'] = clean_me_df['Functional'].replace(dict(Typ=6, Min1=4, Min2=4, Mod=3, Maj1=2, Maj2=2, Sev=1, Sal=0))
    # GarageFinish
    clean_me_df['GarageFinish'] = clean_me_df['GarageFinish'].replace(dict(Fin=2, RFn=1, Unf=0, NA=0))
    # PavedDrive
    clean_me_df['PavedDrive'] = clean_me_df['PavedDrive'].replace(dict(Y=2, P=1, N=0))

    clean_me_df.to_csv(os.path.join(excluded_dir, 'eval_models_5_' + 'mid_clean.csv'), index=False)

    
    # engineered features
    # total bathroom count
    clean_me_df['BathroomCount'] =  clean_me_df['BsmtFullBath'] + (clean_me_df['BsmtHalfBath'] * .5) + clean_me_df['FullBath'] + (clean_me_df['HalfBath'] * .5)



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

sendtofile(excluded_dir, 'X_train.csv', X_train, verbose=True)


# test - use to evaluate model performance
X_test = combined[combined['TRAIN']==0].copy()
X_test.drop('TRAIN', axis=1, inplace=True)

# sub - use to submit the final answer for the kaggle cometition
X_sub = combined[combined['TRAIN']==-1].copy()
X_sub.drop('TRAIN', axis=1, inplace=True)

#all cols
train_cols = ['LotFrontage','LotArea','Utilities','Condition1','Condition2','OverallQual','OverallCond','MasVnrArea','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','HouseAge','RemodelAge','NewRemodel','GarageAge','NewGarage','MSSubClass_20','MSSubClass_30','MSSubClass_40','MSSubClass_45','MSSubClass_50','MSSubClass_60','MSSubClass_70','MSSubClass_75','MSSubClass_80','MSSubClass_85','MSSubClass_90','MSSubClass_120','MSSubClass_150','MSSubClass_160','MSSubClass_180','MSSubClass_190','MSZoning_C (all)','MSZoning_FV','MSZoning_RH','MSZoning_RL','MSZoning_RM','Street_Grvl','Street_Pave','LotShape_IR1','LotShape_IR2','LotShape_IR3','LotShape_Reg','LandContour_Bnk','LandContour_HLS','LandContour_Low','LandContour_Lvl','LotConfig_Corner','LotConfig_CulDSac','LotConfig_FR2','LotConfig_FR3','LotConfig_Inside','LandSlope_Gtl','LandSlope_Mod','LandSlope_Sev','Neighborhood_Blmngtn','Neighborhood_Blueste','Neighborhood_BrDale','Neighborhood_BrkSide','Neighborhood_ClearCr','Neighborhood_CollgCr','Neighborhood_Crawfor','Neighborhood_Edwards','Neighborhood_Gilbert','Neighborhood_IDOTRR','Neighborhood_MeadowV','Neighborhood_Mitchel','Neighborhood_NAmes','Neighborhood_NPkVill','Neighborhood_NWAmes','Neighborhood_NoRidge','Neighborhood_NridgHt','Neighborhood_OldTown','Neighborhood_SWISU','Neighborhood_Sawyer','Neighborhood_SawyerW','Neighborhood_Somerst','Neighborhood_StoneBr','Neighborhood_Timber','Neighborhood_Veenker','BldgType_1Fam','BldgType_2fmCon','BldgType_Duplex','BldgType_Twnhs','BldgType_TwnhsE','HouseStyle_1.5Fin','HouseStyle_1.5Unf','HouseStyle_1Story','HouseStyle_2.5Fin','HouseStyle_2.5Unf','HouseStyle_2Story','HouseStyle_SFoyer','HouseStyle_SLvl','RoofStyle_Flat','RoofStyle_Gable','RoofStyle_Gambrel','RoofStyle_Hip','RoofStyle_Mansard','RoofStyle_Shed','RoofMatl_ClyTile','RoofMatl_CompShg','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','RoofMatl_Tar&Grv','RoofMatl_WdShake','RoofMatl_WdShngl','Exterior1st_AsbShng','Exterior1st_AsphShn','Exterior1st_BrkComm','Exterior1st_BrkFace','Exterior1st_CBlock','Exterior1st_CemntBd','Exterior1st_HdBoard','Exterior1st_ImStucc','Exterior1st_MetalSd','Exterior1st_Plywood','Exterior1st_Stone','Exterior1st_Stucco','Exterior1st_VinylSd','Exterior1st_Wd Sdng','Exterior1st_WdShing','Exterior2nd_AsbShng','Exterior2nd_AsphShn','Exterior2nd_Brk Cmn','Exterior2nd_BrkFace','Exterior2nd_CBlock','Exterior2nd_CmentBd','Exterior2nd_HdBoard','Exterior2nd_ImStucc','Exterior2nd_MetalSd','Exterior2nd_Other','Exterior2nd_Plywood','Exterior2nd_Stone','Exterior2nd_Stucco','Exterior2nd_VinylSd','Exterior2nd_Wd Sdng','Exterior2nd_Wd Shng','MasVnrType_BrkCmn','MasVnrType_BrkFace','MasVnrType_None','MasVnrType_Stone','Foundation_BrkTil','Foundation_CBlock','Foundation_PConc','Foundation_Slab','Foundation_Stone','Foundation_Wood','Heating_Floor','Heating_GasA','Heating_GasW','Heating_Grav','Heating_OthW','Heating_Wall','GarageType_2Types','GarageType_Attchd','GarageType_Basment','GarageType_BuiltIn','GarageType_CarPort','GarageType_Detchd','GarageType_NA','SaleType_COD','SaleType_CWD','SaleType_Con','SaleType_ConLD','SaleType_ConLI','SaleType_ConLw','SaleType_New','SaleType_Oth','SaleType_WD','SaleCondition_Abnorml','SaleCondition_AdjLand','SaleCondition_Alloca','SaleCondition_Family','SaleCondition_Normal','SaleCondition_Partial','Electrical','BathroomCount','IsRegularLotShape','IsLandLevel','IsLandSlopeGntl','IsElectricalSBrkr','IsPavedDrive','IsGarageDetached','HasShed','Neighborhood_Good']




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



if False:
    nan_data = X_train[train_cols].isnull().sum().sort_values(ascending=False)
    print('\n\ndouble check the NaN(s)')
    print(nan_data[nan_data > 0]/len(X_train[train_cols]))
    print("\n\n")


model_lr.fit(X_train[train_cols], y_train)
train_score_lm=model_lr.score(X_train[train_cols], y_train)

# Random forrest
if False:
    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    print("\nheres your pram random_grid")
    print(random_grid)
    print("\n")


    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor(random_state=9261774)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, random_state=9261774, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train[train_cols], y_train)

    print('\n\nBEST ESTIMATOR')
    print (rf_random.best_estimator_ )
    print('\n\nBEST PARAMS')
    print (rf_random.best_params_ )
    print('\n\nBEST SCORE')
    print (rf_random.best_score_ )
    print("\n\n")
    print(1/0)
else:
    #{'n_estimators': 1200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
    model_rf = RandomForestRegressor(random_state=9261774, n_estimators=1200, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=None, bootstrap=False)
    model_rf.fit(X_train[train_cols], y_train)
    train_score_rf=model_rf.score(X_train[train_cols], y_train)
    test_score_lm=model_lr.score(X_test[train_cols], y_test)
    test_score_rf=model_rf.score(X_test[train_cols], y_test)





print('lm training score     : ', train_score_lm)
print('lm test score         : ', test_score_lm)

print('rf training score     : ', train_score_rf)
print('rf test score         : ', test_score_rf)


#GradientBoostingRegressor
# set to True if you want to play with optimization
# set to False when ready to run
if False:
    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(3, 103, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    learning_rate = [0.01, 0.025, 0.05, 0.10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf}
    print("\nheres your param random_grid")
    print(random_grid)
    print("\n")


    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    gb = GradientBoostingRegressor(random_state=9261774)
    #print(gb.get_params().keys())
    

    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    gb_random = RandomizedSearchCV(estimator = gb, param_distributions = random_grid, n_iter = 100, cv = 3,random_state=9261774, n_jobs = -1)
    # Fit the random search model
    gb_random.fit(X_train[train_cols], y_train)

  

    print('\n\nBEST ESTIMATOR')
    print (gb_random.best_estimator_ )
    print('\n\nBEST PARAMS')
    print (gb_random.best_params_ )
    print('\n\nBEST SCORE')
    print (gb_random.best_score_ )
    print("\n\n")
    print(1/0)    


else:
    # {'learning_rate': 0.05, 'max_depth': 4, 'max_features': 1.0, 'min_samples_leaf': 5, 'n_estimators': 221}
    # 0.855131319451837
    # {'learning_rate': 0.05, 'max_depth': 4, 'max_features': 0.25, 'min_samples_leaf': 5, 'n_estimators': 279}
    # 0.86858088329903
    # {'learning_rate': 0.025, 'max_depth': 5, 'max_features': 0.25, 'min_samples_leaf': 5, 'n_estimators': 301}
    # 0.8718522939695451

    # {'learning_rate': 0.025, 'max_depth': 5, 'max_features': 0.25, 'min_samples_leaf': 5, 'n_estimators': 319}
    # 0.8737418819950483
    #BEST PARAMS
    #{'n_estimators': 2000, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 3, 'learning_rate': 0.05}
    model_gb = GradientBoostingRegressor(random_state=9261774,learning_rate=0.05, max_depth=5,min_samples_split=2, max_features="sqrt", min_samples_leaf=2, n_estimators=2000)
    model_gb.fit(X_train[train_cols], y_train)
    #print("Feature Importances")
    #model_gb_fi = model_gb.feature_importances_
    #print (model_gb_fi) 
    
    #
    # feats = {} # a dict to hold feature_name: feature_importance
    # for feature, importance in zip(X_train[train_cols], model_gb.feature_importances_):
    #     feats[feature] = importance #add the name/value pair 
    # importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    # sendtofile(excluded_dir,'importances.csv',importances, verbose=True)

    important_features = pd.Series(data=model_gb.feature_importances_,index=X_train[train_cols].columns)
    important_features.sort_values(ascending=False,inplace=True)
   
    pd.DataFrame(important_features).to_csv(os.path.join(excluded_dir, 'eval_models_5_' + 'important_features.csv'), index=True)
    





 

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

# at this point, who is my worst performer?
# gb naked              :  0.14286
# lr naked              :
# rf naked              :  


# with reduced cols
#lm training score     :  0.8620923437395487
#lm test score         :  0.7973413280413619
#rf training score     :  0.9813814843665337
#rf test score         :  0.8333387647243606
#gb training score     :  0.9691449480278986
#gb test score         :  0.8488831553891795
# kaggle score         :  0.13191 * best yet.  2286/5771


# after the rework, about a dozen categorized vars, a dozen or so one hot encoded
# and a small number of engineered features

#lm training score     :  0.9276846001055292
#lm test score         :  0.8202419903039461
#rf training score     :  0.9999999988909597
#rf test score         :  0.8320806286361078
#gb training score     :  0.9823046833398068
#gb test score         :  0.8591227090039709
#kaggle score          :  0.12913 * best yet!!!!!


# add a few features, hasshed, garage detacthed, etc...
#lm training score     :  0.9279986240136897
#lm test score         :  0.8163988006927535
#rf training score     :  0.9999999992716783
#rf test score         :  0.8352411778378424
#gb training score     :  0.981876656214923
#gb test score         :  0.8510200830718281

# added good neighborhood flag
#lm training score     :  0.9279986240136897
#lm test score         :  0.8163988006926306
#rf training score     :  0.999999998212071
#rf test score         :  0.8432474979151797
#gb training score     :  0.9834801325072335
#gb test score         :  0.8753089279837877


# re did the grid search to tak einto account the new features!
#lm training score     :  0.9279986240136897
#lm test score         :  0.8163988006926306
#rf training score     :  0.999999998212071
#rf test score         :  0.8432474979151797
#gb training score     :  0.9999557988562655  ?over fitting per chance???
#gb test score         :  0.8686661725525482