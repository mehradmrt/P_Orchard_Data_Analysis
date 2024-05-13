#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_swp = pd.read_json('../results/pistachio_SWP.json')
df_cwsi = pd.read_json('../results/pistachio_cwsi.json')
df_td_vpd  = pd.read_json('../results/pistachio_td_vpd.json')
df_im = pd.read_json('../results/pistachio_im_indexes.json')
df_lt = pd.read_json('../results/pistachio_leaftemp.json')
df_ram_c = pd.read_json('../results_cleaned/pistachio_raman_c.json')
df_visnir_c = pd.read_json('../results_cleaned/pistachio_visnir_c.json')
df_TRHP = pd.read_json('../results/pistachio_TRHP.json')

mdf = pd.DataFrame()

tdays_all = ['T1','T2','T3','T4','T5','T6','T7']
tdays_r = ['T5','T6','T7']
treenum = 18

#%%
##### MDF All test days #####
def image_edit(df,arg):
    df = df[df['spec_idx']==arg]
    df = df[['median']]
    df.rename(columns={'median': arg },inplace=True)
    df.reset_index(inplace=True,drop=True)
    return df

ndvi = image_edit(df_im,'NDVI')
gndvi= image_edit(df_im,'GNDVI')
osavi = image_edit(df_im,'OSAVI')
lci  = image_edit(df_im,'LCI')
ndre  = image_edit(df_im,'NDRE')

def swp_class(swp,qvals):
    swp['SWPc'] = swp.loc[:, 'SWP']
    qs = swp.quantile(qvals)
    q1 = qs.iloc[0,0]
    q2 = qs.iloc[1,0]
    swp['SWPc'].mask((swp['SWP'] < q1) ,'WL-1',inplace=True )
    swp['SWPc'].mask((swp['SWP'] >= q1) & (swp['SWP'] < q2),'WL-2' ,inplace=True)
    swp['SWPc'].mask((swp['SWP'] >= q2),'WL-3',inplace=True )

    # bins = (3,7,10,15)
    # group_names = ['WL1','WL2','WL3']
    # main['SWPc2'] = pd.cut(main['SWP'], bins=bins , labels=group_names)
    # main['SWPc2'].isnull()
    return swp

# plt.scatter(mdf.index,mdf['SWP'])
swp = swp_class(df_swp[['SWP']],[.33,.66])
cwsi = df_cwsi.rename(columns={'cwsi': 'CWSI'})
cwsi = cwsi[['CWSI']]

td_vpd = df_td_vpd.rename(columns={'leaf_temp': 'T_c'})
td_vpd = td_vpd[['VPD','T_c']]   

weth = df_TRHP

dfs = [weth,td_vpd,cwsi,ndvi,gndvi,osavi,lci,ndre,swp]
mdf = pd.concat(dfs, axis=1)
mdf.to_json('../results_cleaned/mdf_all.json')

# %%
#### MDF2
#### MDF ram
def ram_tpose(df):
    mdf = pd.DataFrame()
    for k in range(len(tdays_r)):
        # print('\n\n\nTest Number ',tdays_r[k],'\n')
        for i in range(treenum):
            # print()
            arr = (df[df['test_number']==tdays_r[k]][df['tree_id']==i+1])
            arr = arr.loc[:,'Dark Subtracted #1']
            arr = arr.reset_index(drop=True)
            mdf[tdays_r[k]+' '+str(i+1)]= arr
    
    mdf = mdf.transpose()
    mdf.columns = [str(col) + '_ram' for col in mdf.columns]
    mdf.reset_index(drop=True,inplace=True)  

    return mdf

def visnir_tpose(df):
    mdf = pd.DataFrame()
    for k in range(len(tdays_r)):
        # print('\n\n\nTest Number ',tdays_r[k],'\n')
        for i in range(treenum):
            # print()
            arr=(df[df['test_number']==tdays_r[k]][df['tree_id']==i+1])
            arr = arr.loc[:,'Reflect. %']
            arr = arr.reset_index(drop=True)
            mdf[tdays_r[k]+' '+str(i+1)]= arr

    mdf = mdf.transpose()
    mdf.columns = [str(col) + '_VNIR' for col in mdf.columns]
    mdf.reset_index(drop=True,inplace=True)  

    return mdf

mdf_ram = ram_tpose(df_ram_c)
mdf_nv = visnir_tpose(df_visnir_c)

mdfn = (mdf.iloc[-54:,:]).reset_index(drop=True)
mdfn2 = [mdf_ram,mdf_nv,mdfn]
mdf2 = pd.concat(mdfn2, axis=1)

mdf2.to_json('../results_cleaned/mdf2_all.json')

# %%
