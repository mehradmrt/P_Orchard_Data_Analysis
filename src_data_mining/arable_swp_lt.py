#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

# df_im = pd.read_json('../results/pistachio_im_indexes.json')

swp_root = 'C:/Users/Students/Box/Research/IoT4ag/Project_ Water Stress/' \
                +'Data Collection/Pistachio/Ground Data'
df_swp = pd.read_csv(swp_root+'/swp.csv')
df_lt = pd.read_csv(swp_root+'/leaf_temp.csv')
# df_lt.to_json('pistachio_leaftemp.json')

df_sap = pd.read_json('pistachio_sap_data.json')
df_weather = pd.read_json('pistachio_weather_data.json')

arable_root = 'C:/Users/Students/Box/Research/IoT4ag'\
    +'/Project_ Water Stress/Data Collection/Pistachio/Arable_P'
df_arable_T10 = pd.read_csv(arable_root+
'/arable___012444___ 2022_06_21 19_18_53__012444_daily_20220930.csv', skiprows=10)
df_arable_T13 = pd.read_csv(arable_root+
'/arable___012429__ 2022_06_21 20_34_39__012429_daily_20220930.csv', skiprows=10)
df_arable_T10.insert(1,"tree_idx",'10')
df_arable_T13.insert(1,"tree_idx",'13')
df_arable_T10.insert(1,"orchard",'Pistachio')
df_arable_T13.insert(1,"orchard",'Pistachio')
arable = pd.concat([df_arable_T10, df_arable_T13], ignore_index=True)
# arable.to_json('pistachio_arable.json')

# testnum = 7
# treenum = 18
# indexnum = 5
# testdic = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
# idxdic = ['NDVI', 'GNDVI', 'OSAVI', 'LCI' ,'NDRE']
# DOY = [158, 172, 186, 194, 207, 214, 224]

# #%%
# ndvi = df_im.loc[df_im['spec_idx'] == idxdic[0]]['median']
# gndvi = df_im.loc[df_im['spec_idx'] == idxdic[1]]['median']
# osavi = df_im.loc[df_im['spec_idx'] == idxdic[2]]['median']
# lci = df_im.loc[df_im['spec_idx'] == idxdic[3]]['median']
# ndre = df_im.loc[df_im['spec_idx'] == idxdic[4]]['median']
# swp_mn = np.array(df_swp['SWP'])
# lt_mn = np.array(df_lt['leaf_temp'])

# dfcsv = {}
# %%
