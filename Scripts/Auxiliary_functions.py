# functions used in different scripts
import pandas as pd
import numpy as np
import os
def Create_map_product_names(path):
    # Create mapping  DF to include the label of right food group: 
    if isinstance(path,str):
        data_trade_og = pd.read_csv(path)
    else:
        data_trade_og = path
    mapping_products= data_trade_og.loc[:,['L1_foodex','Food_group']].drop_duplicates().sort_values(by='L1_foodex').reset_index(drop=True)
    mapping_products

    mapping_products['fg_short'] = ['Grain products','Vegetable products','Roots & tuberb products','Legumes, nuts, oilseeds & spice',
                                    'Fruit products','Meat products','Milk and dairy products','Egg products',
                                    'Sugar and confections','Animal & vegetable fats','Coffee, cocoa and infusions','Alcoholic beverages']

    mapping_products.set_index('L1_foodex',inplace=True)
    return mapping_products


