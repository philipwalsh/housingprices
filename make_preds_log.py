#################################################
# title   : housing prices: advanced regression techniques
# from    : kaggle.com
# file    : make_preds_log.py
#         : philip walsh
#         : philipwalsh.ds@gmail.com
#         : 2019-12-21
# testing to see if log(SalePrice) helps the performance
# added script name variable so i dont have to worry about hard coding that in
# the sendto file or the opeing and closing print blocks
# if i copy/paste the file to start another file
#
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from os.path import isfile, join
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import lightgbm as lgbm

current_script = os.path.basename(__file__)
log_prefix = os.path.splitext(current_script)[0].replace('_','-')



bVerbose = False
bSaveIntemediateWork = True

my_test_size=0.1    # when ready to make the final sub
#my_test_size=0.20   # normal training eval

working_dir=os.getcwd()
excluded_dir = os.path.join(working_dir, 'excluded') # working_dir + '\excluded'

print('\n\n')
print('*****')
print('***** start of script: ', log_prefix)
print('*****')
print('\n\n')

if bVerbose:
    print('\nworking dir   :', working_dir)


#print('saving X_sub ...', sendtofile(excluded_dir,'submisison_data(cleaned).csv',X_sub))
def sendtofile(outdir, filename, df, verbose=False):
    script_name = log_prefix + '_'
    out_file = os.path.join(outdir, script_name + filename)
    if verbose:
        print("saving file :", out_file)
    df.to_csv(out_file, index=False)
    return out_file


def get_decade(in_val):
    return int(in_val / 10) * 10

def CleanData(clean_me_df):


    #plt.boxplot(clean_me_df['GrLivArea'])
    #plt.show()
    #print(1/0)
    #looks to be a lot of outliers over 4000, lets not use them for training

    # get rid of the Id column
    clean_me_df.drop('Id', axis=1, inplace=True)





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


    clean_me_df['ExterQual'].fillna('TA', inplace=True)

    #These are partially NaN  
    #FireplaceQu     0.486468
    clean_me_df['FireplaceQu'].fillna('NA', inplace=True)
    
    #LotFrontage     0.166495
    LotFrontage_mean = 0
    LotFrontage_mean = clean_me_df['LotFrontage'].mean()
    clean_me_df['LotFrontage'].fillna(LotFrontage_mean, inplace=True)
    #clean_me_df['LotFrontage'] = clean_me_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


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

    #clean_me_df['MSZoning'] = clean_me_df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

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
    #clean_me_df['Exterior1st'] = clean_me_df['Exterior1st'].fillna(clean_me_df['Exterior1st'].mode()[0])
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



    if False:
        nan_data = clean_me_df.isnull().sum().sort_values(ascending=False)
        print('\n\ndouble check the NaN(s)')
        print(nan_data[nan_data > 0]/len(clean_me_df))
        print("\n\n")


        nans = lambda df: df[df.isnull().any(axis=1)]
        print(nans(clean_me_df))
        print(1/0)
        # deal with NaNs

    # a bunch of the categoricals have the same values
    # so we can just loop through them and deal with them almost in bulk
    x = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
    
    # want numbers here, insted of text
    for n in x:
        clean_me_df[n] = clean_me_df[n].replace(dict(Ex=5, Gd=4, TA=3, Fa=2, NA=1, Po=0))
    
    #LightGBM does not like categorical so i turn them to int, and give them a new name
    for n in x:
        clean_me_df[n + '_INT'] = clean_me_df[n].astype('int')

    # then i drop the original
    for n in x:
        clean_me_df.drop(n, axis=1, inplace=True)
    

    # create a feature that adds up all the Quality and Condition vars
    clean_me_df['Total_Qual_Cond'] = (clean_me_df['ExterQual_INT']+clean_me_df['ExterCond_INT']+clean_me_df['BsmtQual_INT']+clean_me_df['BsmtCond_INT']+clean_me_df['HeatingQC_INT']+clean_me_df['KitchenQual_INT']+clean_me_df['FireplaceQu_INT']+clean_me_df['GarageQual_INT']+clean_me_df['GarageCond_INT'])

    # drop the original quality and condition vars
    x = ['ExterQual_INT', 'ExterCond_INT', 'BsmtQual_INT', 'BsmtCond_INT', 'HeatingQC_INT', 'KitchenQual_INT', 'FireplaceQu_INT', 'GarageQual_INT', 'GarageCond_INT']
    for n in x:
        clean_me_df.drop(n, axis=1, inplace=True)

    # handle the one off categoricals that have their own unique values
    # byt replacing text keys with numbers, making an INT col and drop the categorical
    #Utilities
    clean_me_df['Utilities'] = clean_me_df['Utilities'].replace(dict(AllPub=3,NoSewr=2, NoSeWa=1, ELO=0))
    clean_me_df['Utilities_INT'] = clean_me_df['Utilities'].astype('int')     #LightGBM does not like categorical so i turn them to int
    clean_me_df.drop('Utilities', axis=1, inplace=True)
    
    #Condition1
    clean_me_df['Condition1'] = clean_me_df['Condition1'].replace(dict(Artery=5, Feedr=4, Norm=3, RRNn=0, RRAn=0, PosN=3, PosA=2, RRNe=0, RRAe=0))
    clean_me_df['Condition1_INT'] = clean_me_df['Condition1'].astype('int')     #LightGBM does not like categorical so i turn them to int
    clean_me_df.drop('Condition1', axis=1, inplace=True)
    
    #Condition2
    clean_me_df['Condition2'] = clean_me_df['Condition2'].replace(dict(Artery=5, Feedr=4, Norm=3, RRNn=0, RRAn=0, PosN=3, PosA=2, RRNe=0, RRAe=0))
    clean_me_df['Condition2_INT'] = clean_me_df['Condition2'].astype('int')     #LightGBM does not like categorical so i turn them to int
    clean_me_df.drop('Condition2', axis=1, inplace=True)
    
    # BsmtExposure
    clean_me_df['BsmtExposure'] = clean_me_df['BsmtExposure'].replace(dict(Gd=3, Av=2, Mn=1, No=0, NA=0))
    clean_me_df['BsmtExposure_INT'] = clean_me_df['BsmtExposure'].astype('int')     #LightGBM does not like categorical so i turn them to int
    clean_me_df.drop('BsmtExposure', axis=1, inplace=True)
    
    # BsmtFinType1
    clean_me_df['BsmtFinType1'] = clean_me_df['BsmtFinType1'].replace(dict(GLQ=5, ALQ=4, BLQ=3, Rec=2, LwQ=1, Unf=0, NA=0))
    clean_me_df['BsmtFinType1_INT'] = clean_me_df['BsmtFinType1'].astype('int')     #LightGBM does not like categorical so i turn them to int
    clean_me_df.drop('BsmtFinType1', axis=1, inplace=True)
    
    # BsmtFinType2
    clean_me_df['BsmtFinType2'] = clean_me_df['BsmtFinType2'].replace(dict(GLQ=5, ALQ=4, BLQ=3, Rec=2, LwQ=1, Unf=0, NA=0))
    clean_me_df['BsmtFinType2_INT'] = clean_me_df['BsmtFinType2'].astype('int')     #LightGBM does not like categorical so i turn them to int
    clean_me_df.drop('BsmtFinType2', axis=1, inplace=True)
    
    # CentralAir
    clean_me_df['CentralAir'] = clean_me_df['CentralAir'].replace(dict(Y=1, N=0))
    clean_me_df['CentralAir_INT'] = clean_me_df['CentralAir'].astype('int')     #LightGBM does not like categorical so i turn them to int
    clean_me_df.drop('CentralAir', axis=1, inplace=True)
    
    # Electrical
    clean_me_df['Electrical'] = clean_me_df['Electrical'].replace(dict(SBrkr=4,FuseA=3, FuseF=2, FuseP=1, Mix=0))
    clean_me_df['Electrical_INT'] = clean_me_df['Electrical'].astype('int')     #LightGBM does not like categorical so i turn them to int
    clean_me_df.drop('Electrical', axis=1, inplace=True)
    
    # Functional
    clean_me_df['Functional'] = clean_me_df['Functional'].replace(dict(Typ=6, Min1=4, Min2=4, Mod=3, Maj1=2, Maj2=2, Sev=1, Sal=0))
    clean_me_df['Functional_INT'] = clean_me_df['Functional'].astype('int')     #LightGBM does not like categorical so i turn them to int
    clean_me_df.drop('Functional', axis=1, inplace=True)

    # GarageFinish
    clean_me_df['GarageFinish'] = clean_me_df['GarageFinish'].replace(dict(Fin=2, RFn=1, Unf=0, NA=0))
    clean_me_df['GarageFinish_INT'] = clean_me_df['GarageFinish'].astype('int')     #LightGBM does not like categorical so i turn them to int
    clean_me_df.drop('GarageFinish', axis=1, inplace=True)
    
    # PavedDrive
    clean_me_df['PavedDrive'] = clean_me_df['PavedDrive'].replace(dict(Y=2, P=1, N=0))
    clean_me_df['PavedDrive_INT'] = clean_me_df['PavedDrive'].astype('int')     #LightGBM does not like categorical so i turn them to int
    clean_me_df.drop('PavedDrive', axis=1, inplace=True)


    #count the bathrooms, then drop the original vars
    clean_me_df['BathroomCount'] =  clean_me_df['BsmtFullBath'] + (clean_me_df['BsmtHalfBath'] * .5) + clean_me_df['FullBath'] + (clean_me_df['HalfBath'] * .5)
    clean_me_df.drop('BsmtFullBath',axis=1, inplace=True)
    clean_me_df.drop('BsmtHalfBath',axis=1, inplace=True)
    clean_me_df.drop('FullBath',axis=1, inplace=True)
    clean_me_df.drop('HalfBath',axis=1, inplace=True)

    # if any column has more than 99.94 % zeros, drop it like its hot
    overfit = []
    for i in clean_me_df.columns:
        counts = clean_me_df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(clean_me_df) * 100 > 99.94:
            overfit.append(i)
    overfit = list(overfit)
    clean_me_df = clean_me_df.drop(overfit, axis=1).copy()

    # done with this function

    return clean_me_df




##
## MAIN SCRIPT START HERE
##

# load the data
train_data = pd.read_csv('excluded/train.csv', low_memory=False)
sub_data = pd.read_csv('excluded/test.csv', low_memory=False)



print('\nbefore')
print(train_data.shape)
print(train_data.head())
if bSaveIntemediateWork:
    sendtofile(excluded_dir,"train_data-before.csv",train_data, verbose=bVerbose)
#train_data = train_data[train_data['GrLivArea'] < 4000]
#train_data = train_data[train_data['LotArea'] < 100000]
#train_data = train_data[train_data['TotalBsmtSF'] < 3000]
#train_data = train_data[train_data['1stFlrSF'] < 2500]
#print(train_data['BsmtFinSF1'].describe())
temp_mean=0
temp_mean = train_data['BsmtFinSF1'].mean()
train_data['BsmtFinSF1'].fillna(temp_mean, inplace=True)    
train_data = train_data[train_data['BsmtFinSF1'] < 2000.00]


print('\nafter')
print(train_data.shape)
print(train_data.head())
if bSaveIntemediateWork:
    sendtofile(excluded_dir,"train_data-after.csv",train_data, verbose=bVerbose)
#print(1/0)
# find the outliers

print('\ntrain_data.shape before       :', train_data.shape)
train_data = train_data[train_data['GrLivArea'] < 4000]
train_data = train_data[train_data['LotArea'] < 100000]
train_data = train_data[train_data['TotalBsmtSF'] < 3000]
train_data = train_data[train_data['1stFlrSF'] < 4000]
train_data = train_data[train_data['BsmtFinSF1'] < 3000]
train_data = train_data[train_data['MasVnrArea'] < 4000]
train_data = train_data[train_data['TotalBsmtSF'] < 5000]
train_data = train_data[train_data['TotRmsAbvGrd'] < 13]
train_data = train_data[train_data['GarageArea'] < 1200]
train_data = train_data[train_data['LotFrontage'] < 300]
print('train_data.shape after       :', train_data.shape)

# stratified shuffle split
# basically stratify the data by Gross Living Area
# get the test and hold-out data to jive with each other, regarding sampling based on these bins
# only useful in the linear model, the boosted/bagged treee based models should do fine with whatever we give them
train_data['living_area_cat'] = pd.cut(
    train_data['GrLivArea'], 
    bins=[0, 500, 1000, 1500, 2000, 2500, np.inf], 
    labels=[1, 2, 3, 4, 5, 6])



#split = StratifiedShuffleSplit(n_splits=1, test_size=my_test_size, random_state=9261774)
#for train_index, test_index in split.split(train_data, train_data['living_area_cat']):
#    X_train = train_data.loc[train_index] # this is the training data
#    X_test = train_data.loc[test_index]   # this is the hold out, the protion of the training i will use for testing


split = StratifiedShuffleSplit(n_splits=1, test_size=my_test_size, random_state=9261774)
for train_index, test_index in split.split(train_data, train_data['living_area_cat']):
    X_train = train_data.iloc[train_index].copy() # this is the training data
    X_test = train_data.iloc[test_index].copy()   # this is the hold out, the protion of the training i will use for testing


# set up the y aka the label
y_train = X_train['SalePrice']
y_test = X_test['SalePrice']

# drop SalePrice from the x vars
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

if bSaveIntemediateWork:
    sendtofile(excluded_dir, 'combined.csv', combined, verbose=bVerbose)

#####
##### Pull the data apart into the proper data frames. Train, Test (aka holdout) and Submission
#####

# train - use to fit models
X_train = combined[combined['TRAIN']==1].copy()
X_train.drop('TRAIN', axis=1, inplace=True)

if bSaveIntemediateWork:
    sendtofile(excluded_dir, 'X_train.csv', X_train, verbose=bVerbose)


# test - use to evaluate model performance
X_test = combined[combined['TRAIN']==0].copy()
X_test.drop('TRAIN', axis=1, inplace=True)

# sub - use to submit the final answer for the kaggle cometition
X_sub = combined[combined['TRAIN']==-1].copy()
X_sub.drop('TRAIN', axis=1, inplace=True)

#all cols

train_cols = ['LotFrontage','LotArea','OverallQual','OverallCond','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','IsRegularLotShape','IsLandLevel','IsLandSlopeGntl','IsElectricalSBrkr','IsGarageDetached','IsPavedDrive','HasShed','Neighborhood_Good','HouseAge','RemodelAge','NewRemodel','GarageAge','NewGarage','MSSubClass_20','MSSubClass_30','MSSubClass_40','MSSubClass_45','MSSubClass_50','MSSubClass_60','MSSubClass_70','MSSubClass_75','MSSubClass_80','MSSubClass_85','MSSubClass_90','MSSubClass_120','MSSubClass_160','MSSubClass_180','MSSubClass_190','MSZoning_C (all)','MSZoning_FV','MSZoning_RH','MSZoning_RL','MSZoning_RM','Street_Grvl','Street_Pave','LotShape_IR1','LotShape_IR2','LotShape_IR3','LotShape_Reg','LandContour_Bnk','LandContour_HLS','LandContour_Low','LandContour_Lvl','LotConfig_Corner','LotConfig_CulDSac','LotConfig_FR2','LotConfig_FR3','LotConfig_Inside','LandSlope_Gtl','LandSlope_Mod','LandSlope_Sev','Neighborhood_Blmngtn','Neighborhood_Blueste','Neighborhood_BrDale','Neighborhood_BrkSide','Neighborhood_ClearCr','Neighborhood_CollgCr','Neighborhood_Crawfor','Neighborhood_Edwards','Neighborhood_Gilbert','Neighborhood_IDOTRR','Neighborhood_MeadowV','Neighborhood_Mitchel','Neighborhood_NAmes','Neighborhood_NPkVill','Neighborhood_NWAmes','Neighborhood_NoRidge','Neighborhood_NridgHt','Neighborhood_OldTown','Neighborhood_SWISU','Neighborhood_Sawyer','Neighborhood_SawyerW','Neighborhood_Somerst','Neighborhood_StoneBr','Neighborhood_Timber','Neighborhood_Veenker','BldgType_1Fam','BldgType_2fmCon','BldgType_Duplex','BldgType_Twnhs','BldgType_TwnhsE','HouseStyle_1.5Fin','HouseStyle_1.5Unf','HouseStyle_1Story','HouseStyle_2.5Fin','HouseStyle_2.5Unf','HouseStyle_2Story','HouseStyle_SFoyer','HouseStyle_SLvl','RoofStyle_Flat','RoofStyle_Gable','RoofStyle_Gambrel','RoofStyle_Hip','RoofStyle_Mansard','RoofStyle_Shed','RoofMatl_CompShg','RoofMatl_Tar&Grv','RoofMatl_WdShake','RoofMatl_WdShngl','Exterior1st_AsbShng','Exterior1st_AsphShn','Exterior1st_BrkComm','Exterior1st_BrkFace','Exterior1st_CBlock','Exterior1st_CemntBd','Exterior1st_HdBoard','Exterior1st_MetalSd','Exterior1st_Plywood','Exterior1st_Stucco','Exterior1st_VinylSd','Exterior1st_Wd Sdng','Exterior1st_WdShing','Exterior2nd_AsbShng','Exterior2nd_AsphShn','Exterior2nd_Brk Cmn','Exterior2nd_BrkFace','Exterior2nd_CBlock','Exterior2nd_CmentBd','Exterior2nd_HdBoard','Exterior2nd_ImStucc','Exterior2nd_MetalSd','Exterior2nd_Plywood','Exterior2nd_Stone','Exterior2nd_Stucco','Exterior2nd_VinylSd','Exterior2nd_Wd Sdng','Exterior2nd_Wd Shng','MasVnrType_BrkCmn','MasVnrType_BrkFace','MasVnrType_None','MasVnrType_Stone','Foundation_BrkTil','Foundation_CBlock','Foundation_PConc','Foundation_Slab','Foundation_Stone','Foundation_Wood','Heating_GasA','Heating_GasW','Heating_Grav','Heating_OthW','Heating_Wall','GarageType_2Types','GarageType_Attchd','GarageType_Basment','GarageType_BuiltIn','GarageType_CarPort','GarageType_Detchd','GarageType_NA','SaleType_COD','SaleType_CWD','SaleType_Con','SaleType_ConLD','SaleType_ConLI','SaleType_ConLw','SaleType_New','SaleType_Oth','SaleType_WD','SaleCondition_Abnorml','SaleCondition_AdjLand','SaleCondition_Alloca','SaleCondition_Family','SaleCondition_Normal','SaleCondition_Partial','Total_Qual_Cond','Condition1_INT','Condition2_INT','BsmtExposure_INT','BsmtFinType1_INT','BsmtFinType2_INT','CentralAir_INT','Electrical_INT','Functional_INT','GarageFinish_INT','PavedDrive_INT','BathroomCount']
# i like to keep the array of columns that i will be training/predicting with.  
# this gives me the chance to pull one out for testing if say a NaN or inf shows up in the process
# and its not so cumbersome to create.  
# open combined.csv, copy the first row, paste it into notepad and replace tab with ','
# dont forget to remove 'TRAIN' column from the list

#####
##### FIT THE MODELS
#####

if bSaveIntemediateWork:
    sendtofile(excluded_dir,"X_train.csv",X_train[train_cols], verbose=bVerbose)
    sendtofile(excluded_dir,"y_train.csv",pd.DataFrame(y_train), verbose=bVerbose)
    sendtofile(excluded_dir,"X_test.csv",X_test[train_cols], verbose=bVerbose)
    sendtofile(excluded_dir,"y_test.csv",pd.DataFrame(y_test), verbose=bVerbose)

X_train[train_cols].fillna(0,inplace=True)
X_test[train_cols].fillna(0,inplace=True)

#nan check here, set to True as needed
if False:
    nan_data = X_train[train_cols].isnull().sum().sort_values(ascending=False)
    print('\n\ndouble check the NaN(s)')
    print(nan_data[nan_data > 0]/len(X_train[train_cols]))
    print("\n\n")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    nans = lambda df: df[df.isnull().any(axis=1)]
    print(nans(X_train[train_cols]))
    print(1/0) # hard stop



###
### Random Forest optimization block
###
if False:  # set this to True so you can find the best params

    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    #n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 6)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt','log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(2, 100, num = 10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 13, 17]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 7 , 11 , 13]
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

    #print('\n\nBEST ESTIMATOR')
    #print (rf_random.best_estimator_ )
    print('\n\nBEST PARAMS')
    print (rf_random.best_params_ )
    #print('\n\nBEST SCORE')
    #print (rf_random.best_score_ )
    #print("\n\n")
    



###
### GradientBoostingRegressor optimization block
###
if False:  # set to True if you want to play with optimization
    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    #n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 6)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt', 'log2']
    # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(3, 103, num = 11)]
    max_depth = [int(x) for x in np.linspace(2, 100, num = 10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10, 13, 21]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 7 , 11]
    learning_rate = [0.01, 0.025, 0.05, 0.10, 0.075]
    # Method of selecting samples for training each tree
   

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

  

    #print('\n\nBEST ESTIMATOR')
    #print (gb_random.best_estimator_ )
    print('\n\nBEST PARAMS')
    print (gb_random.best_params_ )
    #print('\n\nBEST SCORE')
    #print (gb_random.best_score_ )
    #print("\n\n")
    print(1/0) # hard stop if you like


#####
##### Create Submission
#####
if True and True:

    ###
    ### Light Gradient Boost
    ###
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 8, 
        'learning_rate': 0.025,
        'verbose': 0,
        'num_leaves': 62}
    n_estimators = 300


    train_data = lgbm.Dataset(X_train[train_cols], label=y_train)
    test_data = lgbm.Dataset(X_test[train_cols], label=y_test)

    model_lgbm = lgbm.train(params, train_data, n_estimators, test_data)

    pred_id = sub_data[['Id']]
    pred_y_lgbm=model_lgbm.predict(X_sub[train_cols])
    pred_lgbm=pd.DataFrame(pred_y_lgbm)
    submission_lgbm = pd.concat([submission_id,pred_lgbm], axis='columns', sort=False)
    submission_lgbm.columns=['Id','SalePrice']
    ### save submission file
    sendtofile(excluded_dir,'predictions_lgbm.csv',submission_lgbm, verbose=bVerbose)

       
    ###
    ### Linear Regression
    ###
    model_lr = LinearRegression(normalize=False)

    ### Fit the model
    model_lr.fit(X_train[train_cols], np.log(y_train))
    ## score the model
    train_score_lm=model_lr.score(X_train[train_cols], np.log(y_train))
    test_score_lm=model_lr.score(X_test[train_cols], np.log(y_test))
    print('lm training score     : ', train_score_lm)
    print('lm test score         : ', test_score_lm)

    ### predict the model
    pred_id = sub_data[['Id']]
    pred_y_lr=np.exp(model_lr.predict(X_sub[train_cols]))
    # tack the saved labes (y's) onto the preds into a data frame
    pred_lr=pd.DataFrame(pred_y_lr, columns=['SalePrice'])
    submission_lr = pd.concat([submission_id,pred_lr], axis='columns', sort=False)
    ### save submission file
    sendtofile(excluded_dir,'predictions-lr.csv',submission_lr, verbose=bVerbose)



    ###
    ### Random Forest
    ###

    # BEST PARAMS
    # note, to get the best params, you need to run the optimization block
    # {'n_estimators': 480, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
    model_rf = RandomForestRegressor(random_state=9261774, n_estimators=400, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=None, bootstrap=False)

    model_rf.fit(X_train[train_cols], y_train)
    train_score_rf=model_rf.score(X_train[train_cols], y_train)

    test_score_rf=model_rf.score(X_test[train_cols], y_test)
    print('rf training score     : ', train_score_rf)
    print('rf test score         : ', test_score_rf)


    ###
    ### Gradient Boost
    ###

    # BEST PARAMS
    # note, to get the best params, you need to run the optimization block
    # {'n_estimators': 1620, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 2, 'learning_rate': 0.025}
    model_gb = GradientBoostingRegressor(random_state=9261774,learning_rate=0.025, max_depth=2,min_samples_split=2, max_features="sqrt", min_samples_leaf=1, n_estimators=1620)
    model_gb.fit(X_train[train_cols], y_train)
    train_score_gb=model_gb.score(X_train[train_cols], y_train)
    test_score_gb=model_gb.score(X_test[train_cols], y_test)
    print('gb training score     : ', train_score_gb)
    print('gb test score         : ', test_score_gb)


    #random forest
    pred_y_rf=model_rf.predict(X_sub[train_cols])
    #tack the saved labes (y's) onto the preds into a data frame
    pred_rf=pd.DataFrame(pred_y_rf, columns=['SalePrice'])
    submission_rf = pd.concat([submission_id,pred_rf], axis='columns', sort=False)
    sendtofile(excluded_dir,'predictions_rf.csv',submission_rf, verbose=bVerbose)
    
    pred_y_gb=model_gb.predict(X_sub[train_cols])
    #tack the saved labes (y's) onto the preds into a data frame
    pred_gb=pd.DataFrame(pred_y_gb, columns=['SalePrice'])
    submission_gb = pd.concat([submission_id,pred_gb], axis='columns', sort=False)
    sendtofile(excluded_dir,'predictions_gb.csv',submission_gb, verbose=bVerbose)



    
    #create some submission files

    # linear regression submission
    submission_lr_rf = pd.concat([submission_lr,pred_rf], axis='columns', sort=False)
    
    # random forrest + gradient boost ensemble, 50/50 split
    submission_rf_gb = pd.concat([submission_rf,pred_gb], axis='columns', sort=False)
    submission_rf_gb.columns=['Id', 'SalePrice_RF','SalePrice_GB']
    # start the rf+gb+light gradient boost submisison file here
    submission_rf_gb_lgbm = pd.concat([submission_rf_gb,pred_lgbm], axis='columns', sort=False) 
    submission_rf_gb['SalePrice']=(+submission_rf_gb['SalePrice_RF']+submission_rf_gb['SalePrice_GB'])/2
    submission_rf_gb.drop('SalePrice_RF', axis=1, inplace=True)
    submission_rf_gb.drop('SalePrice_GB', axis=1, inplace=True)
    sendtofile(excluded_dir,'predictions_rf_gb.csv',submission_rf_gb, verbose=bVerbose)

    #lr + rf + gradient boost
    submission_lr_rf_gb = pd.concat([submission_lr_rf,pred_gb], axis='columns', sort=False)
    # start the lr+rf+gb+light gradeitn boost
    submission_lr_rf_gb_lgbm = pd.concat([submission_lr_rf_gb,pred_lgbm], axis='columns', sort=False) 

    submission_lr_rf_gb.columns=['Id','SalePrice_LR', 'SalePrice_RF','SalePrice_GB']
    
    submission_lr_rf_gb['SalePrice']=(submission_lr_rf_gb['SalePrice_LR']+submission_lr_rf_gb['SalePrice_RF']+submission_lr_rf_gb['SalePrice_GB'])/3
    submission_lr_rf_gb.drop('SalePrice_LR', axis=1, inplace=True)
    submission_lr_rf_gb.drop('SalePrice_RF', axis=1, inplace=True)
    submission_lr_rf_gb.drop('SalePrice_GB', axis=1, inplace=True)
    sendtofile(excluded_dir,'predictions_lr_rf_gb.csv',submission_lr_rf_gb, verbose=bVerbose)

    submission_rf_gb_lgbm.columns=['Id', 'SalePrice_RF','SalePrice_GB', 'SalePrice_LGBM']
    submission_rf_gb_lgbm['SalePrice']=(submission_rf_gb_lgbm['SalePrice_RF']+submission_rf_gb_lgbm['SalePrice_GB']+submission_rf_gb_lgbm['SalePrice_LGBM'])/3
    submission_rf_gb_lgbm.drop('SalePrice_RF', axis=1, inplace=True)
    submission_rf_gb_lgbm.drop('SalePrice_GB', axis=1, inplace=True)
    submission_rf_gb_lgbm.drop('SalePrice_LGBM', axis=1, inplace=True)
    sendtofile(excluded_dir,'predictions_rf_gb_lgbm.csv',submission_rf_gb_lgbm, verbose=bVerbose)


    submission_lr_rf_gb_lgbm.columns=['Id','SalePrice_LR', 'SalePrice_RF','SalePrice_GB', 'SalePrice_LGBM']
    # weight the submisisons 10/20/20/50 giving least to lr and most to lgbm
    submission_lr_rf_gb_lgbm['SalePrice']=(
        submission_lr_rf_gb_lgbm['SalePrice_LR'] * 0.10 +
        submission_lr_rf_gb_lgbm['SalePrice_RF'] * 0.10 +
        submission_lr_rf_gb_lgbm['SalePrice_GB'] * 0.40 + 
        submission_lr_rf_gb_lgbm['SalePrice_LGBM'] * 0.40
        )
    sendtofile(excluded_dir,'predictions_lr_rf_gb_lgbm(pre-finalized).csv',submission_lr_rf_gb_lgbm, verbose=bVerbose)
    submission_lr_rf_gb_lgbm.drop('SalePrice_LR', axis=1, inplace=True)
    submission_lr_rf_gb_lgbm.drop('SalePrice_RF', axis=1, inplace=True)
    submission_lr_rf_gb_lgbm.drop('SalePrice_GB', axis=1, inplace=True)
    submission_lr_rf_gb_lgbm.drop('SalePrice_LGBM', axis=1, inplace=True)
    sendtofile(excluded_dir,'predictions_lr_rf_gb_lgbm.csv',submission_lr_rf_gb_lgbm, verbose=bVerbose)


print('\n\n')
print('*****')
print('***** end of script: ', log_prefix)
print('*****')
print('\n\n')





# project wrap up #2
# at the conclusion of this project my best kaggle score is 0.12578 
# I used various resources along the way, mostly books and website tips and tricks from others that have scored well on this kaggle competition
#
# Techniques used in this challenge
# EDA  #i admit, i was very eager to jump into this one so after reading the data description file i didnt spend much time with EDA, too eager to get to the modelling
#   boxplot the GrLivArea (outliers))
# 
# Data cleanup
#   outlier detection and removal 
#   one hot encoding
#   categorization
#   light feature engineering, count bathrooms, add up overall conditoon and quality
#   feature reduction based on % of np.zeros
#
# Modeling
#   because the SalePrice was not normally distributed, for this script only
#       I fit on the log(SalePrice) for the linear regressions, trying to eek out a few more points on the
#       kaggle leader board
#   Stratified splitting based on GrLivArea to be sure i had reprsentative data in training and hold out
#   Random Forest with optimization via grid search
#   Gradient Boosting with optimization via grid search
#   Light Gradient Boosting
#   Ensembling the 4 models with weights for final submission
#
# Conclusion
#   great fun for regression type challenge.  
#   The categorical features (there were many) posed special challenges
#       one hot encoding generated a lot of vars.
#       light gradient boosting did not like categorical and i coudlnt get the param categorical_feature=my_cats to play nice with my categoricals
#   I believe I could spend a few more days adding in more models such as SVM, RVM, KNN to the ensemble
#   and a few more features in order to improve the kaggle score but I think the model is useful at the 
#   moment and I need to move on to bigger and better challenges
#################################################