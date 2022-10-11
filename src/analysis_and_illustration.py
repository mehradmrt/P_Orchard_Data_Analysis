#%%
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

df_im = pd.read_json('Result_Pistachio.json')

swp_root = 'C:/Users/Students/Box/Research/IoT4ag/Project_ Water Stress/' \
                +'Data Collection/Pistachio/Ground Data'
df_swp = pd.read_csv(swp_root+'/swp.csv')

testnum = 7
treenum = 18
indexnum = 5
testdic = ['T1', 'T2', 'T3', 'T4','T5','T6','T7']


# %%

##### Indexes ALL season
def idx_plot(idx_type):
    for i in range(testnum):
        data = df_im[df_im['spec_idx']==idx_type][df_im['test_number'] == testdic[i]]['pixel_array']
        newidx = data.index
        for j in range(treenum):
            arr = data[newidx[j]]
            mean = statistics.median(arr)
            average = sum(arr)/len(arr)
            plt.scatter(i,mean)
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


##### SWP average of all season
def swp_extract():
    for i in range(testnum):
        data = df_swp[df_swp['test_number']==testdic[i]]['SWP']
        newidx = data.index
        # for j in range(treenum):
        #     arr = data[newidx[j]]
        #     avav = sum(arr)/len(arr)
        #     plt.scatter(i,arr)

        avav = sum(data)/len(data)
        plt.scatter(i+1,avav)
    plt.title('SWP Average All Trees')
    plt.show
swp = swp_extract()


#%%
#### Results Per Tree 
def pertree(idx_type,i):
    mn = list([])
    avg_im = list([])
    std = list([])
    swp_av = list([])
    swp_std = list([])

    swp = df_swp[df_swp['tree_idx']==i]
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
# dataset = pd.read_csv('results5by5.csv',header=0,
#                       skipinitialspace=True).values
### Machine Learning
def xdata(idx_type):
    data = df_im[df_im['spec_idx']==idx_type]['pixel_array']
    newidx = data.index
    result = list([])
    for j in range(len(data)):
        
        val = statistics.median(data[newidx[j]])
        result.append(val)
    return result

x = np.array(xdata('NDVI')).astype(float)
y = df_swp['SWP'].to_numpy().astype(float)
