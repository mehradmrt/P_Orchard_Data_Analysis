#%%
import pandas as pd
import numpy as np

df_im = pd.read_json('../results/pistachio_im_indexes.json')

df_swp = pd.read_json('../results/pistachio_SWP.json')
df_lt = pd.read_json('../results/pistachio_leaftemp.json')

df_sap = pd.read_json('../results/pistachio_sap_data.json')
df_weather = pd.read_json('../results/pistachio_weather_data.json')

# df_visnir = pd.read_json('../results/pistachio_visnir.json')
# df_raman = pd.read_json('../results/pistachio_raman.json')

df_arable = pd.read_json('../results/pistachio_arable.json')

Dict = {'T1': '2022-06-07', 'T2': '2022-06-21', 'T3': '2022-07-05', 'T4': '2022-07-13', \
            'T5': '2022-07-26', 'T6': '2022-08-02', 'T7': '2022-08-12'}

#%%
###### Temp capture #####
df= df_weather
df['Date and Time'] = pd.to_datetime(df['Date and Time'])
df3 = df[df['Station ID']=='Weather 3']
df4 = df[df['Station ID']=='Weather 4']

df3max = df3.groupby([df3['Date and Time'].dt.date])['Temperature [℃]'].max()
df4max = df4.groupby([df4['Date and Time'].dt.date])['Temperature [℃]'].max()
df3max = df3max.to_frame()
df4max = df4max.to_frame()
df3max.index = df3max.index.map(str)
df4max.index = df4max.index.map(str)
# df3max.plot.scatter(x=0, y=3,rot=90)

#%%
### Values at noon ###
dict_dates = pd.to_datetime(list(Dict.values()))

df3_filtered_time = df3[(df3['Date and Time'].dt.date.isin(dict_dates.date)) & (df3['Date and Time'].dt.time >= pd.to_datetime('12:30').time()) & (df3['Date and Time'].dt.time <= pd.to_datetime('13:30').time())]
df4_filtered_time = df4[(df4['Date and Time'].dt.date.isin(dict_dates.date)) & (df4['Date and Time'].dt.time >= pd.to_datetime('12:30').time()) & (df4['Date and Time'].dt.time <= pd.to_datetime('13:30').time())]

def find_closest_to_time(df, target_time='13:00:00'):
    target_timestamp = pd.Timestamp(target_time)
    def time_difference(row, target):
        current_time = row['Date and Time'].time()
        current_timestamp = pd.Timestamp.combine(pd.to_datetime('today').date(), current_time)
        return abs((current_timestamp - target).total_seconds() / 60)
    closest_indices = df.groupby(df['Date and Time'].dt.date).apply(lambda x: x.apply(time_difference, args=(target_timestamp,), axis=1).idxmin())
    
    return df.loc[closest_indices]

# df3_noon = find_closest_to_time(df3_filtered_time)
df4_noon = find_closest_to_time(df4_filtered_time)

#%%
def calculate_vpd(row):
    T = row['Temperature [℃]']
    RH = row['Humidity [RH%]']
    VPD = 0.61078 * np.exp((17.27 * T) / (237.3 + T)) * (1 - RH / 100)
    return VPD

# df3_noon['VPD'] = df3_noon.apply(calculate_vpd, axis=1)
df4_noon['VPD [KPa]'] = df4_noon.apply(calculate_vpd, axis=1)
inv_Dict = {pd.to_datetime(v).date(): k for k, v in Dict.items()}
df4_noon['test_number'] = df4_noon['Date and Time'].dt.date.map(inv_Dict)

#%%
### Temperature difference and VPD ###
df_td_vpd = df_lt[['tree_idx','test_number','leaf_temp']]
df_td_vpd.insert(3,'T_noon','')
df_td_vpd.insert(4,'T_diff','')
df_td_vpd.insert(5,'VPD','')

for _, row in df4_noon.iterrows():
    test_number = row['test_number']
    VPD = row['VPD [KPa]']
    Ta = row['Temperature [℃]']  # Assuming this is the correct column for air temperature
    
    # Update the corresponding rows in df_cwsi
    df_td_vpd.loc[df_td_vpd['test_number'] == test_number, 'VPD'] = VPD
    df_td_vpd.loc[df_td_vpd['test_number'] == test_number, 'T_noon'] = Ta

df_td_vpd['T_diff'] = df_td_vpd['leaf_temp'] - df_td_vpd['T_noon'].astype(float)
df_plt = pd.concat([df_td_vpd, df_swp['SWP']], axis=1)

df_td_vpd.to_json('../results/pistachio_td_vpd.json')

#%%
##### Dates DF Creation CWSI #####
def dfcreate(daynum):
    df = pd.DataFrame([],columns=['Dates'])
    for i in Dict:
        temp = pd.date_range(end=Dict[i], periods=daynum)
        temp = temp.to_frame(index=False, name='Dates')
        df = pd.concat([df,temp],ignore_index=True)
    return df

df_dates = dfcreate(1)
# df_dates['Dates'] = df_dates['Dates'].dt.date
df_dates['Dates'] = df_dates['Dates'].dt.strftime('%Y-%m-%d')

def Tfinder(dfT,dfD):
    for i in dfD['Dates']:
        dfT.loc[i]
        print(dfT.loc[i])

Tfinder(df4max,df_dates)

#%%
# df_swp['tree_idx']=df_swp['tree_idx'].add(1)
df_cwsi = df_lt[['tree_idx','test_number','leaf_temp','STD']]
df_cwsi.insert(4,'Ta','')
df_cwsi.insert(5,'cwsi','')
for i in Dict:
    print(i)
    print(df4max.loc[Dict[i]])
    df_cwsi.loc[df_cwsi[df_cwsi['test_number']==i].index,'Ta']= df4max.loc[Dict[i]]['Temperature [℃]']

for i in df_cwsi.index:
    Tc = df_cwsi.iloc[i][2]
    Ta = df_cwsi.iloc[i][4]
    Tcl = df_cwsi['leaf_temp'].min()
    Tcu = df_cwsi['leaf_temp'].max()
    df_cwsi.loc[i,'cwsi'] = ((Tc-Ta)-(Tcl-Ta))/((Tcu-Ta)-(Tcl-Ta))

# df_cwsi.to_json('../results/pistachio_cwsi.json')
#%%


   
# %%
