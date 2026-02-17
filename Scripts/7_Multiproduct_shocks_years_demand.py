# Copy of Multiprod_shock_years to run the demand code first: 
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import matplotlib.ticker as mtick
from joblib import Parallel, delayed
from time import time
import os, glob
from pathlib import Path
import shutil
from matplotlib.lines import Line2D 
from scipy.interpolate import interp1d

#%%

# FUNCTIONS GOOD


def reset_folder(path):
    '''Resets the folder by deleting its contents and creating it again.'''
    print('path to reset:',path)
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
        print('Overwritten savefiles in :',path)
    else: 
        print('Creating new saving directory:', path)
    p.mkdir(parents=True, exist_ok=True)

def fill_na_spline_fit(df, order=1):

    # Create interpolator with linear extrapolation
    years_nan = df['Year'].loc[df['Value'].isna()].unique()
    df_nan = df.copy(deep=True)
    df_nan.loc[df_nan.Year.isin(years_nan),'Year'] =np.nan

    # Get index positions and values that are not NaN

    f = interp1d(df_nan['Year'].dropna(), df_nan['Value'].dropna(), kind='linear', fill_value='extrapolate')
    extrap_values = f(years_nan)

    df.loc[df['Year'].isin(years_nan),'Value'] = extrap_values
    #df['Value'] = df['Value'].interpolate(method='spline',order =order,limit_direction='both',) #order =1,
    return df

def Prep_GDP_Data(GDP, trade, prod,unique_ISO,save_figs=False):
    '''
    Prepares GDP data for the analysis.
    GDP data is filtered to the years of interest and countries with trade and production data.
    Missing GDP data is fixed by interpolating the data.
    '''
    # Load iso codes
    countries_FAOSTAT = pd.read_csv('../Data/input_2023/FAOSTAT_list_countries_ISO.csv',na_filter=False)
    #save end year BELUX
    end_F15 = int(countries_FAOSTAT.loc[countries_FAOSTAT['ISO3 Code']=='F15','End Year'].values[0])
    countries_FAOSTAT=countries_FAOSTAT.loc[:,['Country','ISO3 Code']]
    mapping = countries_FAOSTAT.set_index("ISO3 Code")["Country"].to_dict()


    GDP_iso = pd.merge(GDP,countries_FAOSTAT, right_on='Country', left_on='Area',how='right')
    GDP_iso.drop(columns=['Country','Element','Unit'],inplace=True)

    # Filter GDP in year range:
    GDP_iso = GDP_iso.loc[(GDP_iso.Year>=1986) & (GDP_iso.Year<=2022),:]
    GDP_iso=GDP_iso.loc[GDP_iso['ISO3 Code'].isin(unique_ISO),:]
    GDP_iso = GDP_iso.rename(columns={'ISO3 Code':'ISO'})

    # Check country-years for which we have GDP data available 
    list_gdp= GDP_iso.groupby(by='ISO',as_index=False).apply(lambda x: len(x),include_groups=False)
    list_gdp.rename(columns={None:'count'},inplace=True)
    list_gdp.sort_values(by='count',ascending=True)

    # Check country-years available in trade and prod data: 
    country_year_pairs = trade.groupby(by=['origin_country_ISO','year'],as_index=False).value.count().rename(columns={'origin_country_ISO':'country'})
    country_year_pairs_destin = trade.groupby(by=['destin_country_ISO','year'],as_index=False).value.count().rename(columns={'destin_country_ISO':'country'})
    country_year_pairs_prod = prod.groupby(by=['ISO','year'],as_index=False).value.count().rename(columns={'ISO':'country'})

    country_year_pairs=pd.concat([country_year_pairs,country_year_pairs_destin,country_year_pairs_prod])
    country_year_pairs.drop(columns=['value'],inplace=True)
    country_year_pairs.drop_duplicates(inplace=True)

    available_pairs=country_year_pairs.groupby(by='country',as_index=False).count().sort_values(by='year',ascending=True)
    
    # Compare available data: 
    compare_available_data= pd.merge(available_pairs,list_gdp, left_on='country',right_on='ISO',how='outer')
    #compare_available_data['count']= compare_available_data['count'].fillna(0)
    excluded_countries= compare_available_data.loc[compare_available_data.ISO.isna(),:]
    compare_available_data= compare_available_data.loc[~compare_available_data.ISO.isna(),:]
    to_fix=compare_available_data.loc[compare_available_data['year']>compare_available_data['count'],:]
    print('Countries with missing GDP data:')
    print(to_fix)

    fig, ax = plt.subplots(2,int(np.ceil(len(to_fix)//2)+1), figsize=(10, 6))
    ax = ax.flatten()  # Make it easier to index

    # Fix missing data:
    for i,country_to_fix in tqdm(enumerate(to_fix.ISO)):
        GDP_fix= GDP_iso.loc[GDP_iso.ISO==country_to_fix,:]
        GDP_fix.sort_values(by='Year',ascending=True)
        # Add rows for years missing
        trade_filt = trade.loc[(trade.origin_country_ISO==country_to_fix) | (trade.destin_country_ISO==country_to_fix),:]
        complete_year_list=pd.Series(sorted(trade_filt.year.unique()),name='Year')
        complete_year_list

        #
        merged_GDP =pd.merge(GDP_fix,complete_year_list,how='outer',on='Year')

        merged_GDP=fill_na_spline_fit(merged_GDP,order=1)

        if save_figs == True:

            ax[i].plot(merged_GDP.Year.astype(int),merged_GDP.Value, color='tab:blue',zorder=1)
            if country_to_fix != 'F228':
                ax[i].plot(merged_GDP.Year[merged_GDP.Area.isna()],merged_GDP.Value[merged_GDP.Area.isna()],color='red',zorder=3)
            ax[i].scatter(merged_GDP.Year[merged_GDP.Area.isna()],merged_GDP.Value[merged_GDP.Area.isna()],color='red',zorder=3)
            ax[i].set_title(mapping[country_to_fix] + ' ('+country_to_fix+')')
            ax[i].set_xlim(int(min(merged_GDP.Year))-0.5,int(max(merged_GDP.Year))+0.5)
            ax[i].set_xticks(np.linspace(int(min(merged_GDP.Year)), int(max(merged_GDP.Year)), 3).astype(int))

            ax[7].set_visible(False)
            ax[0].set_ylabel('GDP (million USD)')
            ax[4].set_ylabel('GDP (million USD)')

        # fill other columns
        merged_GDP['Area'] = merged_GDP['Area'].fillna(merged_GDP.Area[~merged_GDP.Area.isna()].unique()[0])
        merged_GDP['ISO']= merged_GDP['ISO'].fillna(merged_GDP.ISO[~merged_GDP.ISO.isna()].unique()[0])
        merged_GDP['Area Code'] = merged_GDP['Area Code'].fillna(merged_GDP['Area Code'][~merged_GDP['Area Code'].isna()].unique()[0])

        # correct gdp_iso
        GDP_iso = GDP_iso.loc[GDP_iso.ISO!=country_to_fix,:]
        GDP_iso = pd.concat([GDP_iso,merged_GDP],ignore_index=True)

    # Merge GDP_iso lux and belgium for the years in which it is BE-LUX (F15):
    merged_BELUX= pd.merge(GDP_iso.loc[(GDP_iso.ISO=='BEL')& (GDP_iso.Year<=end_F15),['Year','Value']],GDP_iso.loc[(GDP_iso.ISO=='LUX')& (GDP_iso.Year<=end_F15),['Year','Value']],on= 'Year',how='outer',suffixes=('_BEL','_LUX'))
    merged_BELUX['Value'] = merged_BELUX['Value_BEL'] + merged_BELUX['Value_LUX']
    merged_BELUX.drop(columns=['Value_BEL','Value_LUX'],inplace=True)
    merged_BELUX[['Area','ISO','Area Code']] = ['Belgium-Luxembourg','F15',np.nan]
    merged_BELUX[['Area','Area Code','Year','Value','ISO']]

    # Remove BEL and LUX from gdp_iso
    GDP_iso = GDP_iso.loc[~((GDP_iso.ISO.isin(['BEL','LUX'])) & (GDP_iso.Year<=end_F15)),:]
    GDP_iso = pd.concat([GDP_iso,merged_BELUX],ignore_index=True)
    # drop F15 from excluded_countries
    excluded_countries = excluded_countries.loc[excluded_countries.country!='F15',:]

    if save_figs == True:
        fig.tight_layout()
        fig.suptitle('Imputation for countries with missing GDP data', fontsize=16,y=1.05)
        legend_elements = [
        Line2D([0], [0], color='tab:blue', lw=2, label='GDP time series'),
        Line2D([0], [0], color='red', lw=2, label='Imputed GDP data')
        ]
        fig.legend(handles=legend_elements, bbox_to_anchor=(1, 0.2), ncol=1, frameon=False)
        fig.savefig(f'../Plots/2023/GDP_missing_data.pdf', bbox_inches='tight')

    return GDP_iso, excluded_countries

def Aggregate_by_food_class (data_trade, data_prod,group_class):

    if group_class=='Food_group':
        save_suffix = 'group'
        # Add group data to production and aggregate it to food groups
        mapping_item_group = data_trade.loc[:,['item_code','L1_foodex','Food_group']].drop_duplicates(ignore_index=True)
        data_prod = pd.merge(data_prod,mapping_item_group, on='item_code',how='left')

        data_prod_filt = data_prod.drop(columns=['item','item_code','area_code','area','element'])
        data_prod_filt = data_prod_filt.groupby(by=['ISO','year','L1_foodex','Food_group','unit'],as_index=False).sum()

        # Aggregate trade data
        data_trade_filt = data_trade.drop(columns=['item','item_code'])
        data_trade_filt = data_trade_filt.groupby(by=['origin_country_ISO','destin_country_ISO','year','L1_foodex','Food_group','unit','origin_country','destin_country'],as_index=False).sum()
    else:
        save_suffix = 'item'
        data_trade_filt= data_trade.copy(deep=True)
        data_prod_filt = data_prod.copy(deep=True)
    return data_trade_filt, data_prod_filt, save_suffix

def Estimate_demand(trade_year, prod_year, pairs_available):
    '''
    Estimate demand for each product-country pair. 
    Also correct negative demand issues from data. 
    '''
    # 
    exp_trade = trade_year.groupby(by=['key_origin'],as_index=False).value.sum()
    imp_trade = trade_year.groupby(by=['key_destin'],as_index=False).value.sum()
    prod = prod_year.groupby(by=['key_prod'],as_index=False).value.sum()

    # Complete index vector
    prod_all = pd.Series(index=pairs_available).fillna(0).astype(int)
    prod_all[prod.key_prod]=prod.value

    exp_all = pd.Series(index=pairs_available).fillna(0).astype(int)
    exp_all[exp_trade.key_origin]=exp_trade.value

    imp_all = pd.Series(index=pairs_available).fillna(0).astype(int)
    imp_all[imp_trade.key_destin]=imp_trade.value
    
    # Estimate demand
    demand = prod_all + imp_all - exp_all

    # demand_correct
    demand_correct = demand.copy(deep=True)
    demand_correct[demand_correct>=0]=0
    demand_correct[demand_correct<0] = -1*demand_correct[demand_correct<0]
    
    # Fix negative demand by adding it to production:
    prod_year_correct = prod_all + demand_correct

    # Re-estimate demand after correction 
    demand = prod_year_correct + imp_all - exp_all

    return demand, prod_year_correct

def Define_shocks(prod_year_all,shock_size):
    shocks = pd.DataFrame({'value':shock_size * prod_year_all[prod_year_all>0].copy(deep=True)})
    shocks['time'] = 1
    shocks = shocks.reset_index().rename(columns={'index':'key'})
    shocks['ISO']= shocks['key'].apply(lambda x: x.split('_')[1])
    # assign a number for each iso
    shocks['shock_group'] = shocks['ISO'].astype('category').cat.codes.astype(int)

    return shocks

def Single_country_shock_parallel(prod_year,trade_year,GDP_year,shock,demand,global_params, simu_params):
    
    country, year, rep = simu_params
    print(year,country, rep)

    # Set seed per repetition
    seed_og = global_params['seed']
    global_params['seed'] = global_params['seed'] + rep if global_params['seed'] is not None else None

    # Sort data and keep only relevant columns
    trade_year = trade_year.loc[:,['L1_foodex','origin_country_ISO','destin_country_ISO','value','key_origin','key_destin','year']].reset_index(drop=True)
    prod_year = prod_year.loc[:,['L1_foodex','key_prod','ISO','value']].reset_index(drop=True)
    GDP_year = GDP_year.loc[:,['Area','Year','ISO','Value']].reset_index(drop=True)

    # Create all pairs of products and countries possible in that year
    ISO_year = list(sorted(set(trade_year.origin_country_ISO.unique()).union(trade_year.destin_country_ISO).union(prod_year.ISO)))
    prod_list = list(sorted(set(trade_year['L1_foodex'].unique()).union(prod_year['L1_foodex'].unique())))

    pairs_available = [i+'_' +j for i in prod_list for j in ISO_year]
    
    if global_params['spread_type'] in ['propTrade'] :
        # Pass the trade preference matrix (W) to the simulation function
        aux_matrix = Trade_preferences(trade_year,pairs_available)
    elif global_params['spread_type'] in ['invGDP']:
        # Define ranking matrix for invGDP
        aux_matrix = GDP_year
    else:  
        # Define rank matrix for invPartners: By total trade
        tot_exports = trade_year.groupby(by = 'origin_country_ISO').value.sum() 
        tot_imports = trade_year.groupby(by = 'destin_country_ISO').value.sum()   
        aux_matrix = pd.Series(index=ISO_year).fillna(0)
        aux_matrix[tot_exports.index] = tot_exports
        aux_matrix[tot_imports.index] = aux_matrix[tot_imports.index] + tot_imports

    # Define single country shocks for all countries:
    shock['value']= shock['value'].astype(float) # convert to float
    shock.loc[:,'value'] = shock.loc[:,'value'] * global_params['shock_size']
    shock.loc[:,'time'] = [1]*len(shock)
    
    list_dd = list()

    # iterate across countries in the shock list (all producers)
    in_trade = trade_year.loc[trade_year.value>0,:]
    og_trade = in_trade.copy(deep=True)
    in_prod = prod_year.set_index('key_prod')['value'] 

    for it in range(0,global_params['max_t']):
        # Proportion of shock spreading to neighbours: inverse to GDP neighbours, dynamically changing. 
        in_trade, available_product= Iteration_simulation(in_trade, in_prod, og_trade ,shock, it, global_params,aux_matrix,demand)
        
        summary_df = available_product.loc[:,['final_dd','failed', 'imports','production','demand']]#,'imports','demand','production']]
        summary_df['ISO_affected'] = summary_df.index
        summary_df['year'] = str(year)
        summary_df['shock_key'] = available_product.index.str.split('_').str[0] + '_'+ country
        summary_df['num_countries_year'] = len(ISO_year)
        summary_df['time'] = it
        summary_df['repetition'] = rep
        
        list_dd.append(summary_df)

    # save for each year
    save_df =  pd.concat(list_dd,ignore_index=True)

    # keep only simulated shocks (filter out cases with no shock)
    save_df = save_df.loc[save_df.shock_key.isin(shock.key_prod),:]
    
    # Find minimum buffer needed to avoid failure (after t>1 failures):
    save_df['failed_nonstart'] = save_df['failed']*(save_df['time']>1) # find failed countries after t=1
    save_df['final_dd_failed_nonstart']= save_df['final_dd']*save_df['failed_nonstart'] # case that only considers countries that failed after t=1

    #save_df['final_dd_failed']= #save_df['final_dd']*save_df['failed'] # case that considers all failed countries (including self failure) 

    # Save single series if required (only for rep1)
    if (rep == 1) and (global_params['save_single'] == True) and (year in [1986, 2022]):
        # save_df in a txt file
        title_save_one = f'../Data/Shock_years/{global_params['folder']}/{global_params['prefix']}One_series_{year}_size_{global_params['shock_size']}_buffer_{global_params['buffer']}_Food_group_{global_params['spread_type']}_seed{seed_og}.txt'
        save_df.to_csv(title_save_one, sep='\t', index=False, mode='a', header=not os.path.exists(title_save_one))

    # Aggregate data to save it:
    sum_failures = save_df.groupby(by=['shock_key','ISO_affected','repetition','year','demand','num_countries_year'],as_index=False).agg(max_dd_failed_nonstart = ('final_dd_failed_nonstart','max'),                                         
                                                                                                                    failed = ('failed','sum'),
                                                                                                                    failed_nonstart= ('failed_nonstart','sum')
                                                                                                                    ).reset_index()
    #Other old saves: sum_dd_all = ('final_dd','sum'),max_dd_all = ('final_dd','max'), sum_dd_failed = ('final_dd_failed','sum'),

    # Define failed
    sum_failures['failed'] = (sum_failures['failed']>0).astype(int) # 1 if ISO_affected is failed at any timepoint, 0 otherwise. 
    sum_failures['failed_nonstart'] = (sum_failures['failed_nonstart']>0).astype(int) 
    sum_failures['shock_key']= sum_failures['shock_key']#+ '_' + sum_failures['year'].astype(str) + '_' + sum_failures['repetition'].astype(str)

    # critical buffer for survival
    sum_failures['buffer_crit']= (sum_failures['max_dd_failed_nonstart']/ sum_failures['demand']).fillna(0)
    sum_failures.drop(columns=['year','max_dd_failed_nonstart','demand'],inplace=True)
    
    # Save data for each year
    title_save = f'../Data/Shock_years/{global_params['folder']}/{global_params['prefix']}All_sum_failures_{year}_size_{global_params['shock_size']}_buffer_{global_params['buffer']}_Food_group_{global_params['spread_type']}_reps{global_params['repetitions']}_seed{seed_og}.txt'
    sum_failures.to_csv(title_save, sep='\t', index=False, mode='a', header=not os.path.exists(title_save))

def Trade_preferences (data_trade,unique_ISO): 
    # UPDATED VERSION OF THE TRADE PREFERENCE ESTIMATION 
    # Define trade matrix
    matrix_trade = data_trade.pivot(index = 'key_origin',columns='key_destin',values='value').fillna(0)
    row_sum= matrix_trade.sum(axis=1) # axis =1 is rownorm
    # replace -1 for rows summing 0 
    row_sum[row_sum==0] = -1
    inv_sum =np.diag(1/row_sum.to_numpy())

    W = np.matmul(inv_sum,matrix_trade.to_numpy())
    W_df = pd.DataFrame(W,columns=matrix_trade.columns,index=matrix_trade.index)

    #W_df[bool_zeros] = 0
    W_all = pd.DataFrame(np.zeros([len(unique_ISO),len(unique_ISO)]), columns=unique_ISO, index=unique_ISO) 
    W_all.update(W_df)

    return W_all

def TradeProp_spread(trade_in, production, demand, W):
    # Estimate total trade from it:
    ex_group = trade_in.groupby(by = 'key_origin').value.sum()
    in_group = trade_in.groupby(by = 'key_destin').value.sum() 
    prod_group = production.groupby(by='key_prod').value.sum()

    # Aggregate data into dataframe: 
    available_product = pd.merge(prod_group, in_group, left_index=True, right_index=True,how='outer',suffixes=['_prod','_import']).fillna(0)
    available_product = pd.merge(available_product, ex_group, left_index=True, right_index=True,how='outer').fillna(0) # exports
    available_product.columns= ['production','imports','exports']
    available_product['demand'] = demand

    # Define estimate of drop in product available: 
    available_product['avail_prod'] = (available_product['production'] + available_product['imports'])
    available_product['expected_dd']  = (available_product['demand'] - available_product['avail_prod'] + available_product['exports'])#.round(n_round)

    # Put shock in matrix form
    shock_prop = np.diag(available_product['expected_dd'])
    shock_propagation =np.matmul(shock_prop, W) 
    shock_propagation.index = available_product.index

    # Put trade in matrix form
    mat_trade = trade_in.pivot(index = 'key_origin',columns='key_destin',values='value').fillna(0)
    data_trade_new = pd.DataFrame(np.zeros([len(W),len(W)]), columns=W.columns, index=W.index)
    data_trade_new.update(mat_trade)

    # Estimate expectedshock spread 
    data_trade_new = data_trade_new - shock_propagation
        
    # Find max possible compensaiton max(Exp_new,0): (equivalent to replacing negative numbers by 0)
    data_trade_new[data_trade_new<0]=0

    # Estimate amount of product recovered by compensation:
    new_exports = data_trade_new.sum(axis = 1)
    available_product['recovered_product'] = available_product['exports']-new_exports
    available_product['final_dd'] = available_product['expected_dd']-available_product['recovered_product']

    # Estimate country failure (accounting for the buffer):  
    available_product['failure_threshold'] = global_params['buffer']*available_product['demand'] 
    available_product['failed']= (available_product['final_dd']>available_product['failure_threshold']).astype(int)
                
    # Transform export data to long format
    data_trade_new= data_trade_new.stack().reset_index()
    data_trade_new.columns = ['key_origin','key_destin','value'] 
    return data_trade_new, available_product

def Stochastic_spread_model(df_trade,og_trade,production, demand, ranking_vector ,global_params):
    '''
    Perfomrs both the 'invGDP' and 'invPartners' methods of stochastic spread.
    '''

    # Estimate total trade from it:
    og_columns = df_trade.columns
    ex_group = og_trade.groupby(by = 'key_origin').value.sum()
    in_group = df_trade.groupby(by = 'key_destin').value.sum() 
    prod_group = production.groupby(by='key_prod').value.sum()

    # Aggregate data into dataframe: 
    available_product = pd.merge(prod_group, in_group, left_index=True, right_index=True,how='outer',suffixes=['_prod','_import']).fillna(0)
    available_product = pd.merge(available_product, ex_group, left_index=True, right_index=True,how='outer').fillna(0) # exports
    available_product.columns= ['production','imports','exports']
    available_product['demand'] = demand

    # Define estimate of drop in product available: 
    available_product['avail_prod'] = (available_product['production'] + available_product['imports'])
    available_product['expected_dd']  = (available_product['demand'] - available_product['avail_prod'] + available_product['exports'])#.round(n_round)

    ranking_vector= ranking_vector.set_index('ISO').Value

    # Rank countries based on inverse GDP:
    df_trade=og_trade.copy(deep=True)
    df_trade = df_trade.loc[df_trade.value>0,:]
    df_trade['variable_2rank']= df_trade.destin_country_ISO.map(ranking_vector)#.fillna(0)
    df_trade['variable_2rank'] = 1/df_trade['variable_2rank'] # inverse GDP

    df_trade['variable_2rank'] = df_trade['variable_2rank'].fillna(0)
    df_trade = df_trade.sort_values(by=['L1_foodex','origin_country_ISO','variable_2rank'],ignore_index=True)
    

    # Define probability of propagation to each country (BUFF_ simulations)
    sum_neighbours=df_trade.groupby(by='key_origin',as_index=False).variable_2rank.sum()
    sum_neighbours.rename(columns={'variable_2rank':'neigh_var2rank'},inplace=True)
    df_trade =pd.merge(df_trade,sum_neighbours, on='key_origin',how='left')
    #df_trade.loc[df_trade.value==0,'variable_2rank']=0 # ignore rows with value=0 for the neighbourhood sum
    df_trade=df_trade.assign(prob=df_trade['variable_2rank'] /df_trade['neigh_var2rank'])
    df_trade.drop(columns=['neigh_var2rank','variable_2rank'],inplace=True)    

    # Make stochastic choice of countries and order df_trade following it
    sample_order = df_trade.groupby(by='key_origin',as_index=True)[['key_destin','prob']].apply(lambda x: list(np.random.choice(x.key_destin,
                                                                                                                                  size=len(x),replace=False, 
                                                                                                                                  p=x.prob))).reset_index().rename(columns={0:'order'}) 
    sample_order['size']=[len(x) for x in  sample_order.order]
    
    # Create ordered key_all: pair of key_origin and key_destin
    origin_repeated= [[sample_order.loc[i,'key_origin']]*sample_order.loc[i,'size'] for i in range(len(sample_order))]
    flattened_origin= [item for sublist in origin_repeated for item in sublist]

    flattened_destin = [item for sublist in sample_order.order for item in sublist]
    ordered_all_key = [flattened_origin[i]+'_'+flattened_destin[i] for i in range(len(flattened_origin))]

    df_trade['key_all']=df_trade['key_origin'] + '_' + df_trade['key_destin']
    
    # Sort dataframe based on 'key_all'
    df_trade['key_all'] = pd.Categorical(df_trade['key_all'],categories=ordered_all_key,ordered=True)
    ordered_df = df_trade.sort_values(by='key_all').reset_index(drop=True)
    ordered_df.drop(columns=['key_all'],inplace=True)

    # Estimate cumsum of exports a country can compensate
    ordered_df['cumsum_val']=ordered_df.groupby(['key_origin'],as_index=True).value.transform(lambda x: x.cumsum().shift(1, fill_value=0))
    
    # Add expected_dd:
    ordered_df = pd.merge(ordered_df,available_product['expected_dd'], left_on='key_origin',right_index=True,how='left')
    
    # Find countries with complete export drops:
    ordered_df['drop_exports'] = ((ordered_df.cumsum_val + ordered_df.value)<=ordered_df.expected_dd).astype(int)

    # Find the first country without a complete drop:
    first_no_fail = ordered_df.groupby(['L1_foodex','origin_country_ISO'],as_index=False)[['drop_exports','cumsum_val','key_origin','destin_country_ISO']].apply(lambda x: First_no_failed(x.loc[x.drop_exports==0,:])).reset_index(drop=True).rename(columns={'cumsum_val':'first_no_fail'})
    first_no_fail['first_no_fail']=1

    ordered_df= pd.merge(ordered_df,first_no_fail.loc[:,['key_origin','destin_country_ISO','first_no_fail']],
             on=['key_origin','destin_country_ISO'],how='left')#.fillna(0)
    
    # Define new exports
    ordered_df['new_exports'] = ordered_df['value']

    # new exports for countries with partial drops
    bool_first_no_fail = ordered_df['first_no_fail']==1
    ordered_df['missing'] =(ordered_df['expected_dd']-ordered_df['cumsum_val'])*bool_first_no_fail

    ordered_df['new_exports']= ordered_df['new_exports'] - (ordered_df['missing'])

    # Define new exports for countries that are completely dropped
    ordered_df.loc[ordered_df.drop_exports==1 ,'new_exports']=0

    # Find amount of product compensted
    new_exp = ordered_df.groupby(['key_origin'],as_index=True).new_exports.sum()
    old_exp = ordered_df.groupby(['key_origin'],as_index=True).value.sum()

    compensated_amount = old_exp-new_exp
    compensated_amount.rename('compensated_amount',inplace=True)
    available_product = pd.merge(available_product,compensated_amount, left_index=True, right_index=True)

    # Find remaining demand unfulfilled: dd' in the article
    available_product['final_dd'] =(available_product['expected_dd']-available_product['compensated_amount'])
    
    # Estimate country failure (consider tolerance/buffer):  
    available_product['failure_threshold'] = global_params['buffer']*available_product['demand'] 
    available_product['failed']= (available_product['final_dd']>available_product['failure_threshold']).astype(int)

    # Drop countries with 0 exports and adapt df format
    new_trade_df = ordered_df.reset_index(drop=True)#.loc[ordered_df.new_exports>0,:].reset_index(drop=True)
    new_trade_df['value']= new_trade_df['new_exports'] # update value
    new_trade_df = new_trade_df.loc[:,og_columns] # keep only useful columns
    return new_trade_df, available_product


def Iteration_simulation(trade_in, data_prod, og_trade, shock_list, it, global_params, aux_matrix, demand):  
    '''
    Function that contains one single iteration of our simulations.
    '''

    # Take extra data
    if global_params['spread_type'] == 'propTrade':
        W = aux_matrix
    else: 
        GDP_year = aux_matrix

    # Apply shock to production
    if all(shock_list.time == it):
        shock_vec = pd.Series(index=data_prod.index).fillna(0)
        shock_vec[shock_list.key_prod] = shock_list.value

        # Substract shock
        production = data_prod - shock_vec
    else:
        
        production = data_prod.copy(deep=True)
    
    production=production.reset_index().rename(columns={'index':'key_prod',0:'value'})
    
    # Propagate the reduction on trade to neighbours:
    if global_params['spread_type'] == 'propTrade':
        # Shock spreads proportional to trade 
        data_trade_new, available_product = TradeProp_spread(trade_in, production, demand, W)
    else:         
        # Shock spreads inversly proportional to GDP
        if global_params['seed'] is not None:
            #Define seed for reproducibility
            np.random.seed(global_params['seed'])
        
        data_trade_new, available_product= Stochastic_spread_model(trade_in, og_trade, production, demand ,GDP_year, global_params)
    
    return data_trade_new.copy(deep=True), available_product 


def order_df (df,sample_order):
    sample_order = sample_order[df.key_origin.unique()[0]]
    df['cat_iso']= pd.Categorical(df.destin_country_ISO,categories=sample_order,ordered=True)
    df = df.sort_values('cat_iso').reset_index(drop=True)
    df.drop(columns=['cat_iso'],inplace=True)
    return df 


def First_no_failed (df):
    df = df.loc[df.cumsum_val ==df.cumsum_val.min(),:]
    
    return df


def Preprocess_data(data_prod_filt,data_trade_filt,GDP_iso,year_list):
    '''
    Filter data input by the ones that also have GDP info for each year. 
    Create all pairs of products and countries possible in that year.
    Estimate demand for each product and country.
    '''
    # Find available pairs of countries and products
    data_trade_filt['key_origin'] = data_trade_filt['L1_foodex']+'_'+data_trade_filt['origin_country_ISO'] 
    data_trade_filt['key_destin'] =  data_trade_filt['L1_foodex']+'_'+ data_trade_filt['destin_country_ISO'] 
    data_prod_filt['key_prod'] = data_prod_filt.loc[:,'L1_foodex']  +'_'+ data_prod_filt['ISO']

    all_prod= pd.DataFrame([])
    all_trade= pd.DataFrame([])
    demand_dict = dict()

    for year in tqdm(year_list):
        # Filter year data 
        trade_year = data_trade_filt.loc[(data_trade_filt.year==year) & (data_trade_filt.unit=='t'),['L1_foodex','origin_country_ISO','destin_country_ISO','value','key_origin','key_destin','year']].reset_index(drop=True)
        prod_year = data_prod_filt.loc[(data_prod_filt.year==year) & (data_prod_filt.unit=='t'),['L1_foodex','unit','ISO','value','key_prod','year']].reset_index(drop=True)
        GDP_year = GDP_iso.loc[GDP_iso.Year==year,['Area','Year','ISO','Value']].reset_index(drop=True)

        # Remove samples with no GDP info: 
        trade_year = trade_year.loc[trade_year.origin_country_ISO.isin(GDP_year.ISO),:]
        trade_year = trade_year.loc[trade_year.destin_country_ISO.isin(GDP_year.ISO),:]
        prod_year = prod_year.loc[prod_year.ISO.isin(GDP_year.ISO),:]

        # Create all pairs of products and countries possible in that year
        ISO_year = list(sorted(set(trade_year.origin_country_ISO.unique()).union(trade_year.destin_country_ISO).union(prod_year.ISO)))
        prod_list = list(sorted(set(trade_year['L1_foodex'].unique()).union(prod_year['L1_foodex'].unique())))

        pairs_available = [i+'_' +j for i in prod_list for j in ISO_year]
        
        # Estimate demand
        demand, prod_year_all= Estimate_demand(trade_year, prod_year, pairs_available)

        new_prod = prod_year_all.reset_index().rename(columns={'index':'key_prod',0:'value'})
        new_prod['year'] = year

        # Update data for simulations
        all_prod = pd.concat([all_prod, new_prod], ignore_index=True)
        all_trade = pd.concat([all_trade, trade_year], ignore_index=True)
        demand_dict[year] = demand

    all_prod['ISO']= all_prod['key_prod'].str.split('_').str[1]
    all_prod['L1_foodex']= all_prod['key_prod'].str.split('_').str[0]

    return all_trade, all_prod , demand_dict


#%% DATA PREPARATION

start_time = time()

group_class='Food_group'# item_code

# Load data
data_trade_og = pd.read_csv('../Data/intermediate/Data_trade_clean.csv')
data_prod_og = pd.read_csv('../Data/intermediate/Data_production_clean.csv')
#%%
# Prepare production data 
#data_prod_og['unit']=data_prod_og['unit'].replace({'t':'tonnes'})
data_prod_og = data_prod_og.rename(columns={'ISO3 Code':'ISO'})
data_prod_og = data_prod_og.loc[data_prod_og.unit=='t',:]

# Get unique countries all data 
unique_ISO = list(sorted(set(data_trade_og.origin_country_ISO.unique()).
                         union(data_trade_og.destin_country_ISO.unique()).
                         union(data_prod_og.ISO.unique())))

# Transform tonnes to kg (avoid float issues):
data_prod_og['value'] = (data_prod_og['value']*1000).astype(int)
data_trade_og['value'] = (data_trade_og['value']*1000).astype(int)

print('Total Countries 1986-2021 before processing:',len(unique_ISO))
print('Total Prods 1986-2021, before processing:',len(data_prod_og.item_code.unique()))

# Group Trade and Prod data: 
parameters = {'group_class':group_class} # to review: in the case of item_code
# Aggregate data 
data_trade_filt, data_prod_filt, save_suffix = Aggregate_by_food_class(data_trade_og, data_prod_og, group_class=parameters['group_class'])

#Load clean GDP data:
GDP_iso= pd.read_csv('../Data/intermediate/GDP_clean.csv')

#%%
# Estimate demand: 
year_list = range(data_trade_filt.year.min(),data_trade_filt.year.max()+1)

data_trade_clean, data_prod_clean, demand_dict = Preprocess_data(data_prod_filt,data_trade_filt,GDP_iso,year_list)

unique_countries = set(data_trade_clean.origin_country_ISO.unique()).union(set(data_trade_clean.destin_country_ISO.unique())).union(set(data_prod_clean.ISO.unique()))
print(f'Total Countries 1986-2022 after processing: {len(unique_countries)}, Total food categories: {len(data_prod_clean.L1_foodex.unique())}')

#%% PARALLEL SIMULATIONS
# Simulation parameters 
max_t= 50
buffer = 0.01 #0.01
shock_size = 1
t_shock = 1
spread_type = 'invGDP' # 'invGDP' / 'propTrade' / 'invPartners' / 'invPartners_outDeg'
unit = 't'
seed = 28 # None (only useful for invGDP)
repetitions = 20 # number of stochastic repetitions
save_single = True # save single series 

# Num cores simulation: 
n_cores = 100 #100
prefix = '2023_'

if spread_type == 'propTrade':
    repetitions = 1
    max_t=100
    folder = 'Multiprod'
    Warning('propTrade is a deterministic method. The seed is only used for invGDP simulations')
elif spread_type == 'invGDP':
    folder = 'Multiprod_stochastic'
else:
    folder = spread_type

print('Number of cores:',n_cores)

global_params = {'max_t':max_t,'shock_size':shock_size,
                       't_shock':t_shock,'buffer':buffer,
                       'unit':unit,'group_class':group_class,
                       'spread_type':spread_type, 'seed': seed,
                       'folder':folder,'repetitions':repetitions,
                       'prefix':prefix,
                       'save_single':save_single} 


# Define list of shocks: 
print('year_list:', len(year_list),'ISO_list:',len(data_prod_clean.ISO.unique()),'reps:',repetitions, 'spread_type',spread_type)

#pairs_shock = [(year,iso) for year in year_list for iso in  data_prod_filt.ISO.unique()] #data_prod_filt.year.unique()
shock_all = data_prod_clean.loc[data_prod_clean.value>0,['key_prod','year','value','ISO']]


shocks_to_do = shock_all.loc[:,['ISO','year']].drop_duplicates(ignore_index=True).sort_values(by=['year','ISO']).reset_index(drop=True)

# Define pairs of shocks to do:
pairs_shock = [(iso,year,rep) for iso,year  in shocks_to_do.itertuples(index=False) for rep in range(1,repetitions+1)] 

#%%

# Reset subfolder
sub_folder = f'{prefix}_size_{shock_size}_buffer_{buffer}_Food_group_{spread_type}_reps{repetitions}_seed{seed}'

reset_folder(f'/home/ariadna/Food_trade/Data/Shock_years/{folder}/{sub_folder}')

global_params['folder'] = f'{folder}/{sub_folder}'

#%% PARALLEL SIMULATION   

Parallel(n_jobs=n_cores)(delayed(Single_country_shock_parallel)(data_prod_clean.loc[(data_prod_clean.year==pair[1])],
                                                                data_trade_clean.loc[(data_trade_clean.year==pair[1])],
                                                                GDP_iso.loc[GDP_iso.Year == pair[1]],
                                                                shock_all.loc[(shock_all.year==pair[1]) & (shock_all.ISO==pair[0]),:],
                                                                demand_dict[pair[1]], global_params, pair) for pair in pairs_shock)

print('start_time:',start_time,'end_time:', time(), 'timedif:', time()-start_time)  
print('year_list:', len(year_list),'ISO_list:',len(data_prod_clean.ISO.unique()),'reps:',repetitions, 'spread_type',spread_type)

#%%# DEBUG SETUP
'''
pair= pairs_shock[0]
prova = Single_country_shock_parallel(data_prod_clean.loc[(data_prod_clean.year==pair[1])],
                                                                    data_trade_clean.loc[(data_trade_clean.year==pair[1])],
                                                                    GDP_iso.loc[GDP_iso.Year == pair[1]],
                                                                    shock_all.loc[(shock_all.year==pair[1]) & (shock_all.ISO==pair[0]),:],
                                                                    demand_dict[pair[1]], global_params, pair)'''


