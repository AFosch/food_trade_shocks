#%% Script to extract the minimal tolerance for each simulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ecdf
from tqdm import tqdm
import seaborn as sns
import pickle
from Auxiliary_functions import *
from joblib import Parallel, delayed
# Load mapping of products

mapping_products = Create_map_product_names('../Data/intermediate/Data_trade_clean.csv')

#%%

# Parameters simulation to run 
nonstart = True # Remove t=1 from simulation to avoid high buffer crit
model = 'invGDP'
reps=20
prefix='2023_'
shock_size= 1
seed = 28
buffer= 0.01

subfolder =f'{prefix}_size_{shock_size}_buffer_{buffer}_Food_group_{model}_reps{reps}_seed{seed}'

# Create all combinations of product, year and fraction avoided
prod_list = list(mapping_products.index)
year_list= list(np.arange(1986,2023,1))
frac_avoided_list= list(np.round(np.arange(0.1,1.1,0.1),2))
all_combinations= [(prod_code,frac, year) for prod_code in prod_list for frac in frac_avoided_list for year in year_list]
#%%
if model == 'invGDP':
    folder = 'Multiprod_stochastic'
elif model == 'propTrade':
    folder = 'Multiprod'
else:
    folder = 'invPartners'

def get_threshold_mean(combination,prefix):
    
    prod_code, frac_avoided, year = combination

    all_data = pd.read_csv(f'../Data/Shock_years/{folder}/{subfolder}/{prefix}All_sum_failures_{year}_size_{shock_size}_buffer_{buffer}_Food_group_{model}_reps{reps}_seed{seed}.txt', sep='\t')
    #all_data['buffer_crit']= (all_data['max_dd_failed']/ all_data['demand']).fillna(0) # buffer crit
    all_data.drop(columns=['index'], inplace=True)

    # Find product data
    prod_data = all_data.loc[(all_data['shock_key'].str.contains(prod_code)) & (all_data['ISO_affected'].str.contains(prod_code)) ].copy()
    prod_data['ISO_shock']= prod_data['shock_key'].str.split('_').str[1]
    prod_data['ISO_receive']= prod_data['ISO_affected'].str.split('_').str[1]
    
    # Define new failed nonstart
    prod_data['failed_nonstart'] = ((prod_data['failed'] == 1) & (prod_data['ISO_shock'] != prod_data['ISO_receive'])).astype(int)
    
    affected_ISO = set(prod_data.ISO_affected.unique())
    # Find the threshold for each affected ISO:
    threshold = pd.Series(index=affected_ISO, data=0,dtype=float)
    for country_prod in list(affected_ISO):
        if nonstart==1:
            data_to_plot = prod_data.loc[(prod_data.failed_nonstart==1) & (prod_data.ISO_affected==country_prod) ,:]#.reset_index(drop=True)
        else: 
            data_to_plot = prod_data.loc[(prod_data.failed==1) & (prod_data.ISO_affected==country_prod),:].reset_index(drop=True)
        
        data_to_plot = data_to_plot.groupby(['ISO_shock','ISO_affected'])['buffer_crit'].mean()
        # find buffer crit that reduces shock_size by X%
        cdf =ecdf(data_to_plot).cdf
        position_thr= np.where(cdf.probabilities>= frac_avoided)[0]

        if len(position_thr) > 0:
            threshold[country_prod] = cdf.quantiles[position_thr[0]]
        else:
            threshold[country_prod] =0 # for the countries that never fail then the 
    
    threshold_append = threshold.reset_index().rename(columns={'index':'ISO',0:'value'})
    threshold_append['product_code'] = mapping_products.loc[prod_code,'fg_short']
    threshold_append['frac_avoided'] = frac_avoided
    threshold_append['year'] = int(year)

    return threshold_append

n_cores=100
list_thresholds = Parallel(n_jobs=n_cores)(delayed(get_threshold_mean)(comb, prefix) for comb in tqdm(all_combinations))

# save as pickle
with open(f'../Data/Shock_years/{folder}/{subfolder}/Thresholds_size_{shock_size}_buffer_{buffer}_Food_group_{model}_reps{reps}_nostart_{nonstart}_seed{seed}.pkl', 'wb') as f:
    pickle.dump(list_thresholds, f)

#%%