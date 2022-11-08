#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

df_im = pd.read_json('pistachio_im_indexes.json')

swp_root = 'C:/Users/Students/Box/Research/IoT4ag/Project_ Water Stress/' \
                +'Data Collection/Pistachio/Ground Data'
df_swp = pd.read_csv(swp_root+'/swp.csv')
df_lt = pd.read_csv(swp_root+'/leaf_temp.csv')

df_sap = pd.read_json('pistachio_sap_data.json')
df_weather = pd.read_json('pistachio_weather_data.json')

arable_root = 'C:/Users/Students/Box/Research/IoT4ag'\
    +'/Project_ Water Stress/Data Collection/Pistachio/Arable_P'
df_arable = pd.read_csv(arable_root+
'/arable___012444___ 2022_06_21 19_18_53__012444_daily_20220930.csv', skiprows=10)

testnum = 7
treenum = 18
indexnum = 5
testdic = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
idxdic = ['NDVI', 'GNDVI', 'OSAVI', 'LCI' ,'NDRE']
DOY = [158, 172, 186, 194, 207, 214, 224]

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
            plt.scatter(DOY[i], median, color='red')
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
    plt.savefig('Pistachio.png')

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
    pertree('GNDVI',i)
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
