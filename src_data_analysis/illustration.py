#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

df_im = pd.read_json('../Results/pistachio_im_indexes.json')
df_swp = pd.read_json('../Results/pistachio_SWP.json')
df_lt = pd.read_json('../Results/pistachio_leaftemp.json')
df_sap = pd.read_json('../Results/pistachio_sap_data.json')
df_weather = pd.read_json('../Results/pistachio_weather_data.json')
df_arable = pd.read_json('../Results/pistachio_arable.json')
df_cwsi = pd.read_json('../results/pistachio_cwsi.json')
df_trhp = pd.read_json('../results/pistachio_TRHP.json')

testnum = 7
treenum = 18
indexnum = 5
testdic = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
idxdic = ['NDVI', 'GNDVI', 'OSAVI', 'LCI' ,'NDRE']
DOY = [158, 172, 186, 194, 207, 214, 224]
orchard = 'Pistachio'

Dict = {'T1': '2022-06-07', 'T2': '2022-06-21', 'T3': '2022-07-05', 'T4': '2022-07-13', \
            'T5': '2022-07-26', 'T6': '2022-08-02', 'T7': '2022-08-12'}

#%%
main = pd.read_json('../results_cleaned/mdf_all.json')
main = main.dropna(how='any',axis=0) 
main.info()
mdf_plt = main[['NDRE','OSAVI','T_c','SWP']]

#%%
#### SWP vs Input ####

def plot_multi_swp_vs_columns(dataframe, x_columns, y_column):
    plt.figure(figsize=(8, 15), dpi=100)
    axis_limits = {'T_c': (20, 50), 'NDRE': (0, 0.3), 'OSAVI': (0.1, 0.5), 'SWP': (0, 20)}
    labels = {'T_c': '$T_c\ [^\circ C]$', 'NDRE': 'NDRE', 'OSAVI': 'OSAVI', 'SWP': '|$\psi$| $_{[bar]}$'}

    for i, x_col in enumerate(x_columns):
        ax = plt.subplot(3, 1, i + 1)
        
        x_data = dataframe[x_col].values
        y_data = dataframe[y_column].values

        model = LinearRegression().fit(x_data.reshape(-1, 1), y_data)
        y_pred = model.predict(x_data.reshape(-1, 1))

        r, _ = pearsonr(x_data, y_data)
        r2 = model.score(x_data.reshape(-1, 1), y_data) 
        rmse = np.sqrt(mean_squared_error(y_data, y_pred))

        plt.scatter(x_data, y_data, color='black', marker='o', alpha=0.5)
        plt.plot(x_data, y_pred, color='black', linewidth=3) 
        
        plt.xlabel(labels[x_col], fontsize=22)
        plt.ylabel(labels[y_column], fontsize=22)

        text_x = ax.get_xlim()[0] + .97 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        text_y = ax.get_ylim()[0] + 1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        plt.text(text_x, text_y, f'$r={r:.2f}$\n$R^2={r2:.2f}$\n$RMSE={rmse:.2f}$', fontsize=16, verticalalignment='top', horizontalalignment='left')
          
        ax.set_xlim(axis_limits[x_col])
        ax.set_ylim(axis_limits[y_column])
        
        plt.grid(True, linestyle='--', linewidth=0.2, color='gray')
        plt.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    plt.savefig('../figures_v1/'+orchard+'_swp_inputs.png',dpi=300)
    plt.show()

x_columns = ['T_c', 'OSAVI', 'NDRE']  
plot_multi_swp_vs_columns(mdf_plt, x_columns, 'SWP')


#%%
##### SWP vs DOY #########
def swp_extract(df_swp, testnum, testdic, DOY, treenum):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)  

    mean_marker = {'marker': 's', 'markersize': 13, 'markerfacecolor': 'black', 'markeredgecolor': 'black'}
    individual_marker_style = {'marker': 'o', 'color': 'gray', 'alpha': 1, 's':100}

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='SWP values'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10, label='Mean ± STD')
    ]

    for i in range(testnum):
        data = df_swp[df_swp['test_number'] == testdic[i]]['SWP']
        mean = np.mean(data)
        std = np.std(data)
        print(mean)
        print(std,'\n')

        for j in range(treenum):
            if j < len(data):
                ax.scatter(DOY[i], data.iloc[j], **individual_marker_style, label='SWP values' if i == 0 and j == 0 else "")

        ax.errorbar(DOY[i], mean, yerr=std, fmt='s', capsize=10, elinewidth=4, color='black', **mean_marker, label='Mean ± STD' if i == 0 else "")

    ax.grid(True, linestyle='--', linewidth=0.2, color='gray')
    ax.set_xlabel('DOY', fontsize=22)
    ax.set_ylabel('|$\psi$|', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_ylim(0, 20)
    ax.legend(handles=legend_handles, fontsize=18)
    plt.tight_layout()
    plt.savefig('../figures_v1/'+orchard+'_swp_doy.png',dpi=600)
    plt.show()

swp_extract(df_swp, testnum, testdic, DOY, treenum)

#%%
#####  weather ######
df= df_weather[df_weather['Station ID']== 'Weather 4']
df['Date and Time'] = pd.to_datetime(df['Date and Time'])

dict_dates = pd.to_datetime(list(Dict.values())).date
df_filtered = df[(df['Date and Time'].dt.date.isin(dict_dates))]

def plot_weather_parameters(df, DOY):
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), dpi=100)
    
    marker_style = {'marker': 'o', 'color': 'black', 'alpha': 0.5}
    line_style = {'linewidth': 2, 'color': 'black'}

    
    date_offsets = pd.date_range(start='2022-01-01', periods=len(DOY), freq='D')
    date_map = {date: offset for date, offset in zip(DOY, date_offsets)}
    df['Plot Date'] = df['Date and Time'].apply(lambda x: date_map[x.date()] + (x - pd.Timestamp(x.date())))

    transition_dates = date_offsets[1:]  

    # Temperature plot
    ax = axes[0]
    ax.scatter(df['Plot Date'], df['Temperature [℃]'], **marker_style)
    ax.plot(df['Plot Date'], df['Temperature [℃]'], **line_style)
    # ax.set_title('Temperature Over Time')
    # ax.set_xlabel('Time')
    ax.set_ylabel('Temperature [℃]', fontsize=22)
    ax.grid(True, linestyle='--', linewidth=0.2)
    ax.set_ylim(0, 50)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xticklabels([])
    for date in transition_dates:
        ax.axvline(x=date, color='black', linestyle='--', linewidth=1) 

    # Humidity plot
    ax = axes[1]
    ax.scatter(df['Plot Date'], df['Humidity [RH%]'], **marker_style)
    ax.plot(df['Plot Date'], df['Humidity [RH%]'], **line_style)
    # ax.set_title('Humidity Over Time')
    # ax.set_xlabel('Time')
    ax.set_ylabel('Relative Humidity [%]',fontsize=22)
    ax.grid(True, linestyle='--', linewidth=0.2)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xticklabels([])
    for date in transition_dates:
        ax.axvline(x=date, color='black', linestyle='--', linewidth=1)  

    # Pressure plot
    ax = axes[2]
    ax.scatter(df['Plot Date'], df['Pressure [hPa]'], **marker_style)
    ax.plot(df['Plot Date'], df['Pressure [hPa]'], **line_style)
    ax.set_xlabel('24 hour Cycle ',fontsize=22)
    ax.set_ylabel('Pressure [hPa]',fontsize=22)
    ax.grid(True, linestyle='--', linewidth=0.2)
    ax.set_ylim(1000, 1015)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xticklabels([])
    for date in transition_dates:
        ax.axvline(x=date, color='black', linestyle='--', linewidth=1)  


    for i, date in enumerate(date_offsets):
        if i < len(DOY):
            for ax in axes:
                ax.text(date + pd.Timedelta(hours=12),  
                        ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.0, 
                        f' {DOY[i]}',  
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        rotation = 20,
                        fontsize=17,
                        color='black')    
                


    plt.tight_layout()
    plt.savefig('../figures_v1/'+orchard+'_weather.png',dpi=600)
    plt.show()

plot_weather_parameters(df_filtered, dict_dates)



# %%
##### Indexes ALL season
def idx_plot(idx_type):
    for i in range(testnum):
        data = df_im[df_im['spec_idx']==idx_type][df_im['test_number'] == testdic[i]]['pixel_array']
        swp_mn = df_swp[df_swp['test_number']==testdic[i]]['SWP']

        newidx = data.index
        swpidx = swp_mn.index
        
        for j in range(treenum):
            arr = data[newidx[j]]
            median = statistics.median(arr)
            plt.scatter(DOY[i], median, color='black')
            # plt.scatter(DOY[i],swp_mn[swpidx[j]], color='blue')
    # plt.figure(str(i+1))
    plt.title(idx_type)
    plt.show

plt.figure()
ndvi = idx_plot('NDVI')
plt.figure()
gndvi = idx_plot('GNDVI')
plt.figure()
osavi = idx_plot('OSAVI')
plt.figure()
lci = idx_plot('LCI')
plt.figure()
ndre = idx_plot('NDRE')

#%%
#### SWP average of all season per day
def swp_extract():
    
    for i in range(testnum):
        data = df_swp[df_swp['test_number']==testdic[i]]['SWP']
        newidx = data.index
        for j in range(treenum):
            col = ['r', 'g','b']
            
            if j in [0,1,2,9,10,11]:
                arr = data[newidx[j]]
                # avav = sum(arr)/len(arr)
                plt.scatter(DOY[i],arr,color=col[0])    

            if j in [6,7,8,12,13,14]:
                arr = data[newidx[j]]
                # avav = sum(arr)/len(arr)
                plt.scatter(DOY[i],arr,color=col[1])

            if j in [3,4,5,15,16,17]:
                arr = data[newidx[j]]
                # avav = sum(arr)/len(arr)
                plt.scatter(DOY[i],arr,color=col[2])

        # avav = sum(data)/len(data)
        # plt.scatter(i,avav)

    plt.title('SWP Average All Trees')
    plt.legend(['stressed','no stress','over watered'])
    plt.xlabel('Day of Year', fontsize = 14)
    plt.ylabel('SWP [bar]', fontsize = 14)
    plt.show

swp = swp_extract()

#%%
#### SWP average of all season per day with Mean and STD
def swp_extract():
    fig, ax = plt.subplots()

    for i in range(testnum):
        data = df_swp[df_swp['test_number']==testdic[i]]['SWP']
        newidx = data.index
        mean = np.mean(data)
        std = np.std(data)
        

        ax.errorbar(DOY[i], mean, std, fmt='s', linewidth=2, ms=10 ,capsize=10, color = 'black')

        for j in range(treenum):
            ax.scatter(DOY[i],data[newidx[j]],color='blue')


    plt.title('Pistachio Orchard', fontsize=16)
    # plt.legend(['stressed','no stress','over watered'])
    plt.xlabel('Day of Year', fontsize = 14)
    plt.ylabel('SWP', fontsize = 14)
    plt.show
    # plt.savefig('Pistachio.png')

swp = swp_extract()

#%%
#### Results Per Tree SWP and Indexes
def pertree(idx_type,i):
    mn = list([])
    avg_im = list([])
    std = list([])
    swp_av = list([])
    swp_std = list([])

    swp = df_swp[df_swp['tree_idx']==i+1]
    swpidx = swp.index

    data0 = df_im[df_im['spec_idx']==idx_type]
    data1 = data0[data0['image_id']==i+1]
    imidx = data1.index
    # print(data1)
    for j in range(len(testdic)):
        arr = data1[data1['test_number']==testdic[j]]['pixel_array'][imidx[j]]
        print(statistics.median(arr))
        mn.append(statistics.median(arr))
        avg_im.append(sum(arr)/len(arr))
        std.append(statistics.stdev(arr))

        swp_av.append(swp[swp['test_number']==testdic[j]]['SWP'][swpidx[j]])
        swp_std.append(swp[swp['test_number']==testdic[j]]['STD'][swpidx[j]])

    # plt.figure()
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    x = [1,2,3,4,5,6,7]
    
    ax1.errorbar(x, mn, std, fmt='o', color ='red' , linewidth=1, capsize=6)
    ax1.plot(x, mn, color = 'red', linewidth=2)
    ax2.errorbar(x, swp_av, swp_std, fmt='o', color ='blue', linewidth=1, capsize=6)
    ax2.plot(x, swp_av, color = 'blue' , linewidth=2)

    ax1.set_xlabel('Test Day Number')
    ax1.set_ylabel('{} Pixels Mean'.format(idx_type), color='r')
    ax2.set_ylabel('SWP Average', color='b')
    plt.title('Tree {}'.format(i+1))
    plt.show()

for i in range(treenum):
    pertree('NDVI',i)
    # pertree('GNDVI')
    # pertree('LCI')
    # pertree('NDRE')


#%%
#### SWP and Leaf Temperature
def swp_lt(i):
    swp_mn = df_swp.loc[df_swp['tree_idx'] == i]['SWP']
    swp_std = df_swp.loc[df_swp['tree_idx'] == i]['STD']

    lt_mn = df_lt.loc[df_lt['tree_idx'] == i]['leaf_temp']
    lt_std = df_lt.loc[df_lt['tree_idx'] == i]['STD']
    print(len(lt_mn))
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    x = testdic
    
    ax1.errorbar(x, lt_mn, lt_std, fmt='o', color ='red' , linewidth=1, capsize=6)
    ax1.plot(x, lt_mn, 'r--', linewidth=2)
    ax2.errorbar(x, swp_mn, swp_std, fmt='o', color ='blue', linewidth=1, capsize=6)
    ax2.plot(x, swp_mn, color = 'blue' , linewidth=2)

    ax1.set_xlabel('Test Day Number')
    ax1.set_ylabel('Leaf Temperature Mean', color='r')
    ax2.set_ylabel('SWP Mean', color='b')
    plt.title('Pistachio Tree {}'.format(i))
    plt.show()


for i in range(treenum):
    swp_lt(i+1)


# %%
#### SWP and Leaf Temperature all trees per test day
def swp_lt(i,ax1,ax2):
    swp_mn = np.array(df_swp.loc[df_swp['test_number'] == testdic[i]]['SWP'])
    lt_mn = np.array(df_lt.loc[df_lt['test_number'] == testdic[i]]['leaf_temp'])

    for j in range(len(lt_mn)):

        x = testdic

        ax1.scatter(x[i], lt_mn[j],color = 'red')
        ax2.scatter(x[i], swp_mn[j],color = 'blue')

        ax1.set_xlabel('Test Day Number')
        ax1.set_ylabel('Leaf Temperature Mean All Trees', color='r')
        ax2.set_ylabel('SWP Mean All Trees', color='b')

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
for i in range(len(testdic)):
    swp_lt(i,ax1,ax2)

plt.title('Pistachio Trees All')    
plt.show()


# %%
####### Leaf_temp Vs SWP All trees

def swp_lt_reg(ax1):
    swp_mn = np.array(df_swp['SWP'])
    lt_mn = np.array(df_lt['leaf_temp'])

    for j in range(len(lt_mn)):

        x = testdic

        ax1.scatter(swp_mn[j], lt_mn[j],color = 'black')
        
        ax1.set_xlabel('SWP [bar]')
        ax1.set_ylabel('Leaf Temperature [C]')

fig, ax1 = plt.subplots()
swp_lt_reg(ax1)

plt.title('Pistachio Trees All')    
plt.show()


# %%
####### Indexes vs SWP All trees
def swp_indexes():
    swp_mn = np.array(df_swp['SWP'])

    for i in range(len(idxdic)):
        idx = df_im.loc[df_im['spec_idx'] == idxdic[i]]['median']
        idxid = idx.index
        fig, ax1 = plt.subplots()

        for j in range(len(swp_mn)):
            ax1.scatter(swp_mn[j], idx[idxid[j]], color = 'blue')
            ax1.set_xlabel('SWP [bar]')
            ax1.set_ylabel('{}'.format(idxdic[i]))        
        
        plt.title('{} vs SWP'.format(idxdic[i]))
        plt.show

swp_indexes()


# %%
####### Indexes vs Leaf Temp All trees
def lt_indexes():
    lt_mn = np.array(df_lt['leaf_temp'])

    for i in range(len(idxdic)):
        idx = df_im.loc[df_im['spec_idx'] == idxdic[i]]['median']
        idxid = idx.index
        fig, ax1 = plt.subplots()

        for j in range(len(lt_mn)):
            ax1.scatter(lt_mn[j], idx[idxid[j]], color = 'red')
            ax1.set_xlabel('Leaf Temperature[C]')
            ax1.set_ylabel('{}'.format(idxdic[i]))        
        
        plt.title('{} vs Leaf Temperature [C]'.format(idxdic[i]))
        plt.show

lt_indexes()


# %%
#%%
#### CWSI average of all season per day
def cwsi_extract():
    
    for i in range(testnum):
        data = df_cwsi[df_cwsi['test_number']==testdic[i]]['cwsi']
        newidx = data.index
        for j in range(treenum):
            col = ['r', 'g','b']
            
            if j in [0,1,2,9,10,11]:
                arr = data[newidx[j]]
                # avav = sum(arr)/len(arr)
                plt.scatter(DOY[i],arr,color=col[0])    

            if j in [6,7,8,12,13,14]:
                arr = data[newidx[j]]
                # avav = sum(arr)/len(arr)
                plt.scatter(DOY[i],arr,color=col[1])

            if j in [3,4,5,15,16,17]:
                arr = data[newidx[j]]
                # avav = sum(arr)/len(arr)
                plt.scatter(DOY[i],arr,color=col[2])

        # avav = sum(data)/len(data)
        # plt.scatter(i,avav)

    plt.title('CWSI All Trees')
    plt.legend(['stressed','no stress','over watered'])
    plt.xlabel('Day of Year', fontsize = 14)
    plt.ylabel('SWP [bar]', fontsize = 14)
    plt.show

swp = cwsi_extract()

# %%
