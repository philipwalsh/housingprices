import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from os.path import isfile, join
from scipy.stats import skew
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


bDisplaySettingsSaved = False
bDataFull = True
bExplore = True
bFit = True
bPredict = True

display_max_rows = 15
display_max_columns = 10
display_width = 120
display_max_colwidth = -1

def display_settings_save():
    bDisplaySettingsSaved = True
    display_max_rows = pd.get_option('display.max_rows')
    display_max_columns = pd.get_option('display.max_columns')
    display_width = pd.get_option('display.width')
    display_max_colwidth = pd.get_option('display.max_colwidth')

def display_settings_restore():
    pd.set_option('display.max_rows', display_max_rows)
    pd.set_option('display.max_columns', display_max_columns)
    pd.set_option('display.width', display_width)
    pd.set_option('display.max_colwidth', display_max_colwidth)

def display_settings_custom(mr, mc, w, mcw):
    
    if (not bDisplaySettingsSaved):
        display_settings_save

    pd.set_option('display.max_rows', mr)
    pd.set_option('display.max_columns', mc)
    pd.set_option('display.width', w)
    pd.set_option('display.max_colwidth', mcw)


#display_settings_custom(None, None, None, -1)
working_dir=os.getcwd()
excluded_dir = os.path.join(working_dir, 'excluded') # working_dir + '\excluded'


def sendtofile(outdir, filename, df):
    script_name="make_preds_2_"
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

    temp_mean = clean_me_df["LotFrontage"].mean()
    clean_me_df['LotFrontage'].fillna(temp_mean, inplace=True)

    temp_mean = clean_me_df["GrLivArea"].mean()
    clean_me_df['GrLivArea'].fillna(temp_mean, inplace=True)


    
    #numeric_feats = clean_me_df.dtypes[clean_me_df.dtypes != "object"].index
    #skewed_feats = clean_me_df[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    #skewed_feats =skewed_feats>0.75
    #skewed_index = skewed_feats.index
    #clean_me_df[skewed_index] = np.log1p(clean_me_df[skewed_index])

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
    train_data = pd.read_csv("excluded/train_full.csv", low_memory=False)
else:
    print('\nLoading small dataset ...')
    train_data = pd.read_csv("excluded/train.csv", low_memory=False)


if bExplore:

    print('\nExplore...\n')
    #print('\ntrain_data.describe()')
    #print(train_data.describe())
    #print('\ntrain_data.info()')
    #print(train_data.info())
    #print("\nobservations")
    #print("we should drop ['Alley', 'MiscFeature', 'PoolQC'] due to too many nulls")
    #print("not sure about Fence yet, i`m 'on the fence'")
    
    #plt.hist(train_data['LotFrontage'])
    #plt.show()


    #print("GarageQual value counts")
    #print(train_data['GarageQual'].value_counts())

    #print("Functional value counts")
    #print(train_data['Functional'].value_counts())


    #print ('\n GrLivArea is most important thing for house prices, lets stratify the training data')
    #plt.hist(train_data['GrLivArea'])
    #plt.show()
    
    #print("\ntrain_data['GrLivArea'].describe()")
    #print(train_data['GrLivArea'].describe())
   
    # looks like we can put the GrLivArea into bins
    # 0, 1000, 2000 , 3000, 4000, 5000 , inf
    train_data['living_area_cat'] = pd.cut(
        train_data['GrLivArea'], 
        bins=[0, 1000,1500,2000,2500, np.inf], 
        labels=[1,2,3,4,5])
    #plt.hist(train_data['living_area_cat'])
    #plt.show()

    train_data.drop('Id', axis=1, inplace=True)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=9261774)
    for train_index, test_index in split.split(train_data, train_data['living_area_cat']):
        train_set = train_data.loc[train_index]
        test_set = train_data.loc[test_index]


    #print('\nbreakdown of the train set by category')
    #print(train_set['living_area_cat'].value_counts()/len(train_set))
    #print('\nbreakdown of the test set by category')
    #print(test_set['living_area_cat'].value_counts()/len(test_set))
    #print("now the holdout/test set is represntative of the training set with respect to gross living area stratified")

    for set_ in (train_set, test_set, train_data):
        set_.drop('living_area_cat', axis=1, inplace=True)

    #print("\nlets try to find correlations, using the corr()")
    #corr_matrix = train_data.corr()
    #print(corr_matrix['SalePrice'].sort_values(ascending=False))

    # of interest, 
    # OverallQual, GrLivArea, GrageCars, GarageArea, TotalBsmtSF, 1stFlrSF, fullBath, TotalRmsAbvGrd, YearBuilt

    #from pandas.plotting import scatter_matrix
    #attribs = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF'] #, '1stFlrSF', 'FullBath', 'TotalRmsAbvGrd', 'YearBuilt']
    #scatter_matrix(train_data[attribs])
    #plt.show()

    clean_df = train_set.copy()
    clean_df = CleanData(clean_df)
    

    #OverallQual+OverallCond
    #from pandas.plotting import scatter_matrix
    #attribs = ['SalePrice', 'OverallQual', 'GrLivArea', 'OverallCond'] #, '1stFlrSF', 'FullBath', 'TotalRmsAbvGrd', 'YearBuilt']
    #scatter_matrix(clean_df[attribs])
    #plt.show()
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    df_desc = pd.DataFrame(clean_df.describe())
    print('saving clean_df_desc ...', sendtofile(excluded_dir,'clean_df_desc.csv',df_desc))


    scrubbed_df = clean_df.copy()
    scrubbed_df = ScrubData(scrubbed_df)

    print("\nlets try to find the new cleaned correlations, using the corr()")
    corr_matrix2 = scrubbed_df.corr()
    print(corr_matrix2['SalePrice'].sort_values(ascending=False))

    #print(scrubbed_df.dtypes)
    #print("\ngot any nas")
    #print(scrubbed_df.isna().any())

    #submission_data = pd.read_csv("excluded/test.csv", low_memory=False)
    #print(submission_data["YearBuilt"].describe())
    # 1910 - 2006

    #print("\ngarage qual")
    #print(clean_df['GarageQual'].describe())

#nas


else:
    print('\nSkipping Explore\n')

    
train_cols = ['OverallQual','GrLivArea','GC_3','1stFlrSF','FullBath','YearRemodAdd','TotRmsAbvGrd','Fireplaces','DECOLD_01','DECOLD_01']

if bFit:
    print('saving scrubbed_df ...', sendtofile(excluded_dir,'scrubbed_df.csv',scrubbed_df))

    train_vars = scrubbed_df.copy()
    train_vars.drop('SalePrice', axis=1, inplace=True)
    train_labels = scrubbed_df['SalePrice']
    model_lin_reg = LinearRegression()
    model_lin_reg.fit(train_vars[train_cols], train_labels)

    some_data = train_vars.iloc[:5]
    some_labels = train_labels.iloc[:5]
    lin_reg_pred = model_lin_reg.predict(some_data[train_cols])
    print("\nsome_labels")
    print(some_labels)
    print("\nlin_reg_pred")
    print(lin_reg_pred)

    preds = model_lin_reg.predict(train_vars[train_cols])
    rmse_ = mean_squared_error(train_labels,preds )
    print('\nLinear Regression RMSE  :', rmse_)


    model_tree_reg = DecisionTreeRegressor()
    tree_reg_pred = model_tree_reg.fit(train_vars[train_cols], train_labels)
    preds = model_tree_reg.predict(train_vars[train_cols])
    rmse_ = mean_squared_error(train_labels,preds )
    print('\nDecision Tree RMSE    :', rmse_)

    model_forrest_reg = RandomForestRegressor()
    model_forrest_reg.fit(train_vars[train_cols], train_labels)
    preds = model_forrest_reg.predict(train_vars[train_cols])
    rmse_ = mean_squared_error(train_labels,preds )
    print('\nRandom Forest RMSE    :', rmse_)
    


else:
    print('\nSkipping Fit\n')

if bPredict:
    print('\nPredict ...\n')
    
    submission_data = pd.read_csv("excluded/test_full.csv", low_memory=False)
    
    submission_y = submission_data[['Id']]
    #?
    submission_data.drop("Id", axis="columns", inplace=True)
    #?

    cleaned_df = submission_data.copy()
    cleaned_df = CleanData(cleaned_df)
    scrubbed_df = cleaned_df.copy()
    scrubbed_df = ScrubData(cleaned_df)

    #print("any nulls?")
    #print(scrubbed_df.isna().any())
    pred_y=model_lin_reg.predict(scrubbed_df[train_cols])
    pred_df=pd.DataFrame(pred_y, columns=['SalePrice'])
    submission_df = pd.concat([submission_y,pred_df], axis="columns", sort=False)
    print("saving submission_df ...", sendtofile(excluded_dir,"lin_reg_preds.csv",submission_df))

else:
    print('\nSkipping predict\n')

