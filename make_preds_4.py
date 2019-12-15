# phil walsh
# pmcphilwalsh@gmail.com
# this script is based off of make_predictions_2.py
# goals for this script
#  1) focus on linear regression
#  2) properly evaluate the model ( look at preds, observed and residuals )
#  3) tackle one hot encoding ( stretch goal )

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

bDataFull = True
bExplore = True
bFit = True
bPredict = False


working_dir=os.getcwd()
excluded_dir = os.path.join(working_dir, 'excluded') # working_dir + '\excluded'


print('\n\n*****')
print('***** start of sript: make_preds_3.py')
print('*****')
print('\nworking dir   :', working_dir)
def sendtofile(outdir, filename, df):
    script_name='make_preds_3_'
    out_file = os.path.join(outdir, script_name + filename) 
    df.to_csv(out_file, index=False)
    return out_file

def get_decade(in_val):
    return int(in_val / 10) * 10

def ScrubData(clean_me_df):
    clean_me_df.drop('Street', axis=1, inplace=True)
    clean_me_df.drop('Utilities', axis=1, inplace=True)
    clean_me_df.drop('Condition2', axis=1, inplace=True)
    clean_me_df.drop('BldgType', axis=1, inplace=True)
    clean_me_df.drop('HouseStyle', axis=1, inplace=True)
    clean_me_df.drop('RoofStyle', axis=1, inplace=True)
    clean_me_df.drop('RoofMatl', axis=1, inplace=True)
    clean_me_df.drop('Exterior1st', axis=1, inplace=True)
    clean_me_df.drop('Exterior2nd', axis=1, inplace=True)
    clean_me_df.drop('MasVnrType', axis=1, inplace=True)
    clean_me_df.drop('ExterQual', axis=1, inplace=True)
    clean_me_df.drop('ExterCond', axis=1, inplace=True)
    clean_me_df.drop('Foundation', axis=1, inplace=True)
    clean_me_df.drop('BsmtQual', axis=1, inplace=True)
    clean_me_df.drop('BsmtCond', axis=1, inplace=True)
    clean_me_df.drop('BsmtExposure', axis=1, inplace=True)
    clean_me_df.drop('BsmtFinType1', axis=1, inplace=True)
    clean_me_df.drop('BsmtFinSF1', axis=1, inplace=True)
    clean_me_df.drop('BsmtFinType2', axis=1, inplace=True)
    clean_me_df.drop('Heating', axis=1, inplace=True)
    clean_me_df.drop('HeatingQC', axis=1, inplace=True)
    clean_me_df.drop('CentralAir', axis=1, inplace=True)
    clean_me_df.drop('KitchenQual', axis=1, inplace=True)
    clean_me_df.drop('Functional', axis=1, inplace=True)
    clean_me_df.drop('FireplaceQu', axis=1, inplace=True)
    clean_me_df.drop('GarageType', axis=1, inplace=True)
    clean_me_df.drop('GarageFinish', axis=1, inplace=True)
    clean_me_df.drop('PavedDrive', axis=1, inplace=True)
    clean_me_df.drop('SaleType', axis=1, inplace=True)
    clean_me_df.drop('SaleCondition', axis=1, inplace=True)
    clean_me_df.drop('GarageYrBlt', axis=1, inplace=True)
    clean_me_df.drop('MasVnrArea', axis=1, inplace=True)
    clean_me_df.drop('BsmtHalfBath', axis=1, inplace=True)
    clean_me_df.drop('GarageArea', axis=1, inplace=True)
    clean_me_df.drop('BsmtFinSF2', axis=1, inplace=True)    
    clean_me_df.drop('BsmtUnfSF', axis=1, inplace=True)        
    clean_me_df.drop('TotalBsmtSF', axis=1, inplace=True)
    clean_me_df.drop('BsmtFullBath', axis=1, inplace=True)
          
    return clean_me_df

def CleanData(clean_me_df):


    clean_me_df.drop('Alley', axis=1, inplace=True)
    clean_me_df.drop('MiscFeature', axis=1, inplace=True)
    clean_me_df.drop('PoolQC', axis=1, inplace=True)
    clean_me_df.drop('Fence', axis=1, inplace=True)

    temp_mean = clean_me_df['LotFrontage'].mean()
    clean_me_df['LotFrontage'].fillna(temp_mean, inplace=True)

    temp_mean = clean_me_df['GrLivArea'].mean()
    clean_me_df['GrLivArea'].fillna(temp_mean, inplace=True)
    
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
    clean_me_df.drop('Condition1', axis=1, inplace=True) 




    clean_me_df['Neighborhood'].fillna('NAmes', inplace=True)    
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'NAmes'), 'NH_NAmes'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'CollgCr'), 'NH_CollgCr'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'OldTown'), 'NH_OldTown'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'Edwards'), 'NH_Edwards'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'Somerst'), 'NH_Somerst'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'Gilbert'), 'NH_Gilbert'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'NridgHt'), 'NH_NridgHt'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'Sawyer'), 'NH_Sawyer'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'NWAmes'), 'NH_NWAmes'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'SawyerW'), 'NH_SawyerW'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'BrkSide'), 'NH_BrkSide'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'Crawfor'), 'NH_Crawfor'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'Mitchel'), 'NH_Mitchel'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'NoRidge'), 'NH_NoRidge'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'Timber'), 'NH_Timber'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'IDOTRR'), 'NH_IDOTRR'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'ClearCr'), 'NH_ClearCr'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'StoneBr'), 'NH_StoneBr'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'SWISU'), 'NH_SWISU'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'MeadowV'), 'NH_MeadowV'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'Blmngtn'), 'NH_Blmngtn'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'BrDale'), 'NH_BrDale'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'Veenker'), 'NH_Veenker'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'NPkVill'), 'NH_NPkVill'] = 1
    clean_me_df.loc[(clean_me_df['Neighborhood'] == 'Blueste'), 'NH_Blueste'] = 1
    clean_me_df['NH_NAmes'].fillna(0, inplace=True)
    clean_me_df['NH_CollgCr'].fillna(0, inplace=True)
    clean_me_df['NH_OldTown'].fillna(0, inplace=True)
    clean_me_df['NH_Edwards'].fillna(0, inplace=True)
    clean_me_df['NH_Somerst'].fillna(0, inplace=True)
    clean_me_df['NH_Gilbert'].fillna(0, inplace=True)
    clean_me_df['NH_NridgHt'].fillna(0, inplace=True)
    clean_me_df['NH_Sawyer'].fillna(0, inplace=True)
    clean_me_df['NH_NWAmes'].fillna(0, inplace=True)
    clean_me_df['NH_SawyerW'].fillna(0, inplace=True)
    clean_me_df['NH_BrkSide'].fillna(0, inplace=True)
    clean_me_df['NH_Crawfor'].fillna(0, inplace=True)
    clean_me_df['NH_Mitchel'].fillna(0, inplace=True)
    clean_me_df['NH_NoRidge'].fillna(0, inplace=True)
    clean_me_df['NH_Timber'].fillna(0, inplace=True)
    clean_me_df['NH_IDOTRR'].fillna(0, inplace=True)
    clean_me_df['NH_ClearCr'].fillna(0, inplace=True)
    clean_me_df['NH_StoneBr'].fillna(0, inplace=True)
    clean_me_df['NH_SWISU'].fillna(0, inplace=True)
    clean_me_df['NH_MeadowV'].fillna(0, inplace=True)
    clean_me_df['NH_Blmngtn'].fillna(0, inplace=True)
    clean_me_df['NH_BrDale'].fillna(0, inplace=True)
    clean_me_df['NH_Veenker'].fillna(0, inplace=True)
    clean_me_df['NH_NPkVill'].fillna(0, inplace=True)
    clean_me_df['NH_Blueste'].fillna(0, inplace=True)
    clean_me_df.drop('Neighborhood', axis=1, inplace=True) 


    #int(value / 10) * 10
    temp_mean = clean_me_df['YearBuilt'].mean()
    clean_me_df['YearBuilt'].fillna(temp_mean, inplace=True)    

    clean_me_df['YearDecade'] = clean_me_df['YearBuilt'].astype(int).map(lambda x: get_decade(x))
    clean_me_df['DecadesOld'] = 2010 - clean_me_df['YearDecade']
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 0), 'DECOLD_00'] = 1
    clean_me_df['DECOLD_00'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 10), 'DECOLD_01'] = 1
    clean_me_df['DECOLD_01'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 20), 'DECOLD_02'] = 1
    clean_me_df['DECOLD_02'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 30), 'DECOLD_03'] = 1
    clean_me_df['DECOLD_03'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 40), 'DECOLD_04'] = 1
    clean_me_df['DECOLD_04'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 50), 'DECOLD_05'] = 1
    clean_me_df['DECOLD_05'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 60), 'DECOLD_06'] = 1
    clean_me_df['DECOLD_06'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 70), 'DECOLD_07'] = 1
    clean_me_df['DECOLD_07'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 80), 'DECOLD_08'] = 1
    clean_me_df['DECOLD_08'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 90), 'DECOLD_09'] = 1
    clean_me_df['DECOLD_09'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 100), 'DECOLD_10'] = 1
    clean_me_df['DECOLD_10'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 110), 'DECOLD_11'] = 1
    clean_me_df['DECOLD_11'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 120), 'DECOLD_12'] = 1
    clean_me_df['DECOLD_12'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 130), 'DECOLD_13'] = 1
    clean_me_df['DECOLD_13'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['DecadesOld'] == 140), 'DECOLD_14'] = 1
    clean_me_df['DECOLD_14'].fillna(0, inplace=True)


    clean_me_df.drop('YearBuilt', axis=1, inplace=True)
    clean_me_df.drop('DecadesOld', axis=1, inplace=True)
    clean_me_df.drop('YearDecade', axis=1, inplace=True)

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
    clean_me_df.drop('Electrical', axis=1, inplace=True) 


    clean_me_df['GarageQual'].fillna('TA', inplace=True)    
    clean_me_df.loc[(clean_me_df['GarageQual'] == 'TA'), 'GARQ_TA'] = 1
    clean_me_df['GARQ_TA'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['GarageQual'] == 'Fa'), 'GARQ_Fa'] = 1
    clean_me_df['GARQ_Fa'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['GarageQual'] == 'Gd'), 'GARQ_Gd'] = 1
    clean_me_df['GARQ_Gd'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['GarageQual'] == 'Po'), 'GARQ_Po'] = 1
    clean_me_df['GARQ_Po'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['GarageQual'] == 'Ex'), 'GARQ_Ex'] = 1
    clean_me_df['GARQ_Ex'].fillna(0, inplace=True)
    clean_me_df.drop('GarageQual', axis=1, inplace=True) 


    clean_me_df['GarageCond'].fillna('TA', inplace=True)    
    clean_me_df.loc[(clean_me_df['GarageCond'] == 'TA'), 'GARC_TA'] = 1
    clean_me_df['GARC_TA'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['GarageCond'] == 'Fa'), 'GARC_Fa'] = 1
    clean_me_df['GARC_Fa'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['GarageCond'] == 'Gd'), 'GARC_Gd'] = 1
    clean_me_df['GARC_Gd'].fillna(0, inplace=True)
    clean_me_df.loc[(clean_me_df['GarageCond'] == 'Po'), 'GARC_Po'] = 1
    clean_me_df['GARC_Po'].fillna(0, inplace=True)
    clean_me_df.drop('GarageCond', axis=1, inplace=True) 


    clean_me_df['LotShape'].fillna('Reg', inplace=True)    
    clean_me_df.loc[(clean_me_df['LotShape'] == 'Reg'), 'LOTS_Reg'] = 1
    clean_me_df.loc[(clean_me_df['LotShape'] == 'IR1'), 'LOTS_IR1'] = 1
    clean_me_df.loc[(clean_me_df['LotShape'] == 'IR2'), 'LOTS_IR2'] = 1
    clean_me_df.loc[(clean_me_df['LotShape'] == 'IR3'), 'LOTS_IR3'] = 1
    clean_me_df['LOTS_Reg'].fillna(0, inplace=True)
    clean_me_df['LOTS_IR1'].fillna(0, inplace=True)
    clean_me_df['LOTS_IR2'].fillna(0, inplace=True)
    clean_me_df['LOTS_IR3'].fillna(0, inplace=True)
    clean_me_df.drop('LotShape', axis=1, inplace=True) 

    clean_me_df['LandContour'].fillna('Lvl', inplace=True)    
    clean_me_df.loc[(clean_me_df['LandContour'] == 'Lvl'), 'LC_Lvl'] = 1
    clean_me_df.loc[(clean_me_df['LandContour'] == 'Bnk'), 'LC_Bnk'] = 1
    clean_me_df.loc[(clean_me_df['LandContour'] == 'HLS'), 'LC_HLS'] = 1
    clean_me_df.loc[(clean_me_df['LandContour'] == 'Low'), 'LC_Low'] = 1
    clean_me_df['LC_Lvl'].fillna(0, inplace=True)
    clean_me_df['LC_Bnk'].fillna(0, inplace=True)
    clean_me_df['LC_HLS'].fillna(0, inplace=True)
    clean_me_df['LC_Low'].fillna(0, inplace=True)
    clean_me_df.drop('LandContour', axis=1, inplace=True) 

    clean_me_df['LotConfig'].fillna('Inside', inplace=True)    
    clean_me_df.loc[(clean_me_df['LotConfig'] == 'Inside'), 'LCFG_Inside'] = 1
    clean_me_df.loc[(clean_me_df['LotConfig'] == 'Corner'), 'LCFG_Corner'] = 1
    clean_me_df.loc[(clean_me_df['LotConfig'] == 'CulDSac'), 'LCFG_CulDSac'] = 1
    clean_me_df.loc[(clean_me_df['LotConfig'] == 'FR2'), 'LCFG_FR2'] = 1
    clean_me_df.loc[(clean_me_df['LotConfig'] == 'FR3'), 'LCFG_FR3'] = 1
    clean_me_df['LCFG_Inside'].fillna(0, inplace=True)
    clean_me_df['LCFG_Corner'].fillna(0, inplace=True)
    clean_me_df['LCFG_CulDSac'].fillna(0, inplace=True)
    clean_me_df['LCFG_FR2'].fillna(0, inplace=True)
    clean_me_df['LCFG_FR3'].fillna(0, inplace=True)
    clean_me_df.drop('LotConfig', axis=1, inplace=True) 

    clean_me_df['LandSlope'].fillna('Gtl', inplace=True)    
    clean_me_df.loc[(clean_me_df['LandSlope'] == 'Gtl'), 'LS_Gtl'] = 1
    clean_me_df.loc[(clean_me_df['LandSlope'] == 'Mod'), 'LS_Mod'] = 1
    clean_me_df.loc[(clean_me_df['LandSlope'] == 'Sev'), 'LS_Sev'] = 1
    clean_me_df['LS_Gtl'].fillna(0, inplace=True)
    clean_me_df['LS_Mod'].fillna(0, inplace=True)
    clean_me_df['LS_Sev'].fillna(0, inplace=True)
    clean_me_df.drop('LandSlope', axis=1, inplace=True) 



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
    clean_me_df.drop('GarageCars', axis=1, inplace=True) 


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
    clean_me_df.drop('MSZoning', axis=1, inplace=True) 


   
    print('cleaned!')
    return clean_me_df

if bDataFull:
    print('\nLoading full dataset ...')
    train_data = pd.read_csv('excluded/train_full.csv', low_memory=False)
else:
    print('\nLoading small dataset ...')
    train_data = pd.read_csv('excluded/train.csv', low_memory=False)




#tweaking the bins didnt get me a better score.  it did get me a diff score, just a hair worse
train_data['living_area_cat'] = pd.cut(
    train_data['GrLivArea'], 
    bins=[0, 1000, 1500, 2000, 2500, np.inf], 
    #bins=[0, 1000, 1500, 2000, 2500, 3000, 4000, np.inf], slightly wors, by a tiny bit
    labels=[1, 2, 3, 4, 5])

train_data.drop('Id', axis=1, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=9261774)
for train_index, test_index in split.split(train_data, train_data['living_area_cat']):
    X_train = train_data.loc[train_index]
    X_test = train_data.loc[test_index]



y_train = X_train['SalePrice']
y_test = X_test['SalePrice']

X_train.drop('SalePrice', axis=1, inplace=True)
X_test.drop('SalePrice', axis=1, inplace=True)


for set_ in (X_train, X_test, train_data):
    set_.drop('living_area_cat', axis=1, inplace=True)



#clean it - fix missing data, encode the categories
X_train = CleanData(X_train)
print('saving cleaned X_train ...', sendtofile(excluded_dir,'X_train(claened).csv',X_train))
#scrub it - remove the vars that havent been cleaned yet (categoricals and vars that contain missing data)
X_train = ScrubData(X_train)
print('saving scrubbed X_train ...', sendtofile(excluded_dir,'X_train(scrubbed).csv',X_train))

#full set
train_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearRemodAdd', '1stFlrSF', 
'2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 
'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 
'MoSold', 'YrSold', 'C1_Norm', 'C1_Feedr', 'C1_PosN', 'C1_Artery', 'C1_RRAe', 'C1_RRNn', 'C1_RRAn', 'C1_PosA', 
'C1_RRNe', 'NH_NAmes', 'NH_CollgCr', 'NH_OldTown', 'NH_Edwards', 'NH_Somerst', 'NH_Gilbert', 'NH_NridgHt', 
'NH_Sawyer', 'NH_NWAmes', 'NH_SawyerW', 'NH_BrkSide', 'NH_Crawfor', 'NH_Mitchel', 'NH_NoRidge', 'NH_Timber', 
'NH_IDOTRR', 'NH_ClearCr', 'NH_StoneBr', 'NH_SWISU', 'NH_MeadowV', 'NH_Blmngtn', 'NH_BrDale', 'NH_Veenker', 
'NH_NPkVill', 'NH_Blueste', 'DECOLD_00', 'DECOLD_01', 'DECOLD_02', 'DECOLD_03', 'DECOLD_04', 'DECOLD_05', 
'DECOLD_06', 'DECOLD_07', 'DECOLD_08', 'DECOLD_09', 'DECOLD_10', 'DECOLD_11', 'DECOLD_12', 'DECOLD_13', 
'DECOLD_14', 'E_SBrkr', 'E_FuseF', 'E_FuseA', 'E_FuseP', 'E_Mix', 'GARQ_TA', 'GARQ_Fa', 'GARQ_Gd', 'GARQ_Po', 
'GARQ_Ex', 'GARC_TA', 'GARC_Fa', 'GARC_Gd', 'GARC_Po', 'LOTS_Reg', 'LOTS_IR1', 'LOTS_IR2', 'LOTS_IR3', 'LC_Lvl', 
'LC_Bnk', 'LC_HLS', 'LC_Low', 'LCFG_Inside', 'LCFG_Corner', 'LCFG_CulDSac', 'LCFG_FR2', 'LCFG_FR3', 'LS_Gtl', 
'LS_Mod', 'LS_Sev', 'GC_1', 'GC_2', 'GC_3', 'GC_4', 'GC_5', 'MSZ_RL', 'MSZ_RM', 'MSZ_C', 'MSZ_FV', 'MSZ_RH']


# removed GC_5 and E_Mix
train_cols = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearRemodAdd', '1stFlrSF', 
'2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 
'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 
'MoSold', 'YrSold', 'C1_Norm', 'C1_Feedr', 'C1_PosN', 'C1_Artery', 'C1_RRAe', 'C1_RRNn', 'C1_RRAn', 'C1_PosA', 
'C1_RRNe', 'NH_NAmes', 'NH_CollgCr', 'NH_OldTown', 'NH_Edwards', 'NH_Somerst', 'NH_Gilbert', 'NH_NridgHt', 
'NH_Sawyer', 'NH_NWAmes', 'NH_SawyerW', 'NH_BrkSide', 'NH_Crawfor', 'NH_Mitchel', 'NH_NoRidge', 'NH_Timber', 
'NH_IDOTRR', 'NH_ClearCr', 'NH_StoneBr', 'NH_SWISU', 'NH_MeadowV', 'NH_Blmngtn', 'NH_BrDale', 'NH_Veenker', 
'NH_NPkVill', 'NH_Blueste', 'DECOLD_00', 'DECOLD_01', 'DECOLD_02', 'DECOLD_03', 'DECOLD_04', 'DECOLD_05', 
'DECOLD_06', 'DECOLD_07', 'DECOLD_08', 'DECOLD_09', 'DECOLD_10', 'DECOLD_11', 'DECOLD_12', 'DECOLD_13', 
'DECOLD_14', 'E_SBrkr', 'E_FuseF', 'E_FuseA', 'E_FuseP', 'GARQ_TA', 'GARQ_Fa', 'GARQ_Gd', 'GARQ_Po', 
'GARQ_Ex', 'GARC_TA', 'GARC_Fa', 'GARC_Gd', 'GARC_Po', 'LOTS_Reg', 'LOTS_IR1', 'LOTS_IR2', 'LOTS_IR3', 'LC_Lvl', 
'LC_Bnk', 'LC_HLS', 'LC_Low', 'LCFG_Inside', 'LCFG_Corner', 'LCFG_CulDSac', 'LCFG_FR2', 'LCFG_FR3', 'LS_Gtl', 
'LS_Mod', 'LS_Sev', 'GC_1', 'GC_2', 'GC_3', 'GC_4', 'MSZ_RL', 'MSZ_RM', 'MSZ_C', 'MSZ_FV', 'MSZ_RH']


# check correlations
if False:
    corr_cols = train_cols.copy()
    corr_cols.append('SalePrice')
    max_rows = pd.get_option('display.max_rows')
    train_data = CleanData(train_data)
    train_data = ScrubData(train_data)
    pd.set_option('display.max_rows', None)
    print("\nlets try to find the current correaltions")
    corr_matrix2 = train_data[corr_cols].corr()
    print(corr_matrix2['SalePrice'].sort_values(ascending=False))
    pd.set_option('display.max_rows', max_rows)

    # positive correlation
    # OverallQual|GrLivArea|GC_3|1stFlrSF|FullBath|TotRmsAbvGrd|YearRemodAdd|Fireplaces|DECOLD_01|NH_NridgHt|LotFrontage|NH_NoRidge|WoodDeckSF
    # negative correlation
    # OverallQual|GrLivArea|GC_3|1stFlrSF|FullBath|TotRmsAbvGrd|YearRemodAdd|Fireplaces|DECOLD_01|NH_NridgHt|LotFrontage|NH_NoRidge|WoodDeckSF
    # GC_1|MSZ_RM|LOTS_Reg|E_FuseA|DECOLD_09|NH_OldTown|NH_NAmes|DECOLD_06

    # lets try just positive
    train_cols = ['OverallQual', 'GrLivArea', 'GC_3', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearRemodAdd', 'Fireplaces', 'DECOLD_01', 'NH_NridgHt', 'LotFrontage', 'NH_NoRidge', 'WoodDeckSF']
    #training score     :  0.8124677506796857
    #test score         :  0.7649712400989457
    #kaggle score       :  0.18317 :(
    #kaggle best        :  0.16576
    # lets try positive and negative
    train_cols = ['OverallQual', 'GrLivArea', 'GC_3', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearRemodAdd', 'Fireplaces', 'DECOLD_01', 'NH_NridgHt', 'LotFrontage', 'NH_NoRidge', 'WoodDeckSF','GC_1', 'MSZ_RM', 'LOTS_Reg', 'E_FuseA', 'DECOLD_09', 'NH_OldTown', 'NH_NAmes', 'DECOLD_06']
    #training score     :  0.8229924404888985
    #test score         :  0.767898780925574
    #kaggle score       :  0.43765 :(:(  WTF!!!?
    #kaggle best        :  0.16576




#train_cols = ['GrLivArea','1stFlrSF','LotArea','MoSold','YearRemodAdd','YrSold','LotFrontage','OpenPorchSF','WoodDeckSF','TotRmsAbvGrd','OverallQual','2ndFlrSF','BedroomAbvGr','OverallCond','MSSubClass']



if False:
    ## check with lasso, and see what it recommends as the importance
    from sklearn.datasets import make_classification
    from sklearn.ensemble import ExtraTreesClassifier
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                random_state=0)

    forest.fit(X_train[train_cols], y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()






#normalize=False
#lm training score     :  0.8690424877980849
#lm test score         :  0.7970224279145512
#rf training score     :  0.9700308018508477
#rf test score         :  0.7802256471579926
#normalize=True

model_lr = LinearRegression(normalize=False)
print('model_lr Parameters currently in use:\n')
print(model_lr.get_params())

model_lr.fit(X_train[train_cols], y_train)
train_score_lm=model_lr.score(X_train[train_cols], y_train)

model_rf = RandomForestRegressor(random_state=9261774)
print('model_rf Parameters currently in use:\n')
print(model_rf.get_params())


model_rf.fit(X_train[train_cols], y_train)
train_score_rf=model_rf.score(X_train[train_cols], y_train)
    



#clean it - fix missing data, encode the categories
X_test = CleanData(X_test)
print('saving cleaned X_test ...', sendtofile(excluded_dir,'X_test(claened).csv',X_test))
#scrub it - remove the vars that havent been cleaned yet (categoricals and vars that contain missing data)
X_test = ScrubData(X_test)
print('saving scrubbed X_test ...', sendtofile(excluded_dir,'X_test(scrubbed).csv',X_test))

test_score_lm=model_lr.score(X_test[train_cols], y_test)
test_score_rf=model_rf.score(X_test[train_cols], y_test)


print('lm training score     : ', train_score_lm)
print('lm test score         : ', test_score_lm)

print('rf training score     : ', train_score_rf)
print('rf test score         : ', test_score_rf)

#happy with above score, create a submission
#load the test data
sub_data = pd.read_csv("excluded/test_full.csv", low_memory=False)

#set up the submission ids, will be used later in a full submission dataframe
pred_id = sub_data[['Id']]

#remove Id from the independent variables
sub_data.drop("Id", axis="columns", inplace=True)

#clean it
sub_data = CleanData(sub_data)
#scrub it
sub_data = ScrubData(sub_data)

#get predictions
pred_y_lr=model_lr.predict(sub_data[train_cols])
#tack the saved labes (y's) onto the preds into a data frame
pred_lr=pd.DataFrame(pred_y_lr, columns=['SalePrice'])
submission_lr = pd.concat([pred_id,pred_lr], axis="columns", sort=False)

print("saving submission_lr ...", sendtofile(excluded_dir,"predictions_lr.csv",submission_lr))



#get predictions
pred_y_rf=model_rf.predict(sub_data[train_cols])
#tack the saved labes (y's) onto the preds into a data frame
pred_rf=pd.DataFrame(pred_y_rf, columns=['SalePrice'])
submission_rf = pd.concat([pred_id,pred_rf], axis="columns", sort=False)

print("saving submission_rf ...", sendtofile(excluded_dir,"predictions_rf.csv",submission_rf))


print('\n\n*****')
print('***** end of sript: make_preds_3.py')
print('*****')


#random forest alone got me to a kaggle score of 0.16853
#Ensembled this time, averaged te linear with he random forest
#lm training score     :  0.8690424877980849
#lm test score         :  0.7970224279145512
#rf training score     :  0.9718557949492583
#rf test score         :  0.7918664368953635
#kaggle best(previous) :  0.16576
#kaggle score(current) :  0.14772 ****NEW BEST****






pipe = Pipeline([('classifier' , RandomForestRegressor())])
# pipe = Pipeline([('classifier', RandomForestClassifier())])

# Create param grid.

#param_grid = [
#    {'classifier' : [LinearRegression()],
#     'classifier__penalty' : ['l1', 'l2'],
#    'classifier__C' : np.logspace(-4, 4, 20),
#    'classifier__solver' : ['liblinear']},
#    {'classifier' : [RandomForestRegressor()],
#    'classifier__n_estimators' : list(range(10,101,10)),
#    'classifier__max_features' : list(range(6,32,5))}
#]

param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear']},
    {'classifier' : [RandomForestRegressor()],
    'classifier__n_estimators' : list(range(10,101,10)),
    'classifier__max_features' : list(range(6,32,5))}
]

# Create grid search object

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)

# Fit on data

best_clf = clf.fit(X_train, y_train)
print(best_clf)

