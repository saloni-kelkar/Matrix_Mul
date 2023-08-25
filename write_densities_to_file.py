import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import gaussian_kde as gaussian_kde

pd.options.mode.chained_assignment = None

full_mcm = pd.read_excel('Data/Whisky_Full.xlsx', usecols='A:IP').to_numpy()

solutes_df = pd.read_excel('Data/MCM_whisky.xlsx', skiprows=2, sheet_name='Features', usecols='A:E', header=None,
                                  names=['I', 'U1', 'U2', 'U3', 'U4'], dtype={'I': np.float64, 'U1': np.float64, 'U2': np.float64, 'U3': np.float64, 'U4': np.float64})
solutes_first_NAN_row = solutes_df[solutes_df.isnull().all(axis=1) == True].index.tolist()[0]
solutes_df = solutes_df.loc[0:(solutes_first_NAN_row-1)]
solutes_df['in_range'] = False

solvents_df = pd.read_excel('Data/MCM_whisky.xlsx', skiprows=2, sheet_name='Features', usecols='H:L', header=None,
                                   names=['J', 'V1', 'V2', 'V3', 'V4'], dtype={'J': np.float64, 'V1': np.float64, 'V2': np.float64, 'V3': np.float64, 'V4': np.float64})
solvents_first_NAN_row = solvents_df[solvents_df.isnull().all(axis=1) == True].index.tolist()[0]
solvents_df = solvents_df.loc[0:(solvents_first_NAN_row-1)]
solvents_df['in_range'] = False

combined_array = np.column_stack((np.outer(solutes_df['U1'], solvents_df['V1']).ravel(), np.outer(solutes_df['U2'], solvents_df['V2']).ravel(),
                         np.outer(solutes_df['U3'], solvents_df['V3']).ravel(), np.outer(solutes_df['U4'], solvents_df['V4']).ravel(), full_mcm.ravel()))

features_df = pd.DataFrame(combined_array, columns=['first', 'second', 'third', 'fourth', 'mcm'])
densities_df = pd.DataFrame()

mcm_arr = features_df['mcm'].to_numpy()
for col in features_df.columns[:4]:
    density_col_name = col + '_density'
    feature_arr = features_df[col].to_numpy()
    features_arr = np.column_stack((feature_arr, mcm_arr))
    kernel = gaussian_kde(features_arr.T)
    density_values = kernel(features_arr.T)
    densities_df[density_col_name] = density_values

densities_df.to_excel('density_columns.xlsx', index=False)