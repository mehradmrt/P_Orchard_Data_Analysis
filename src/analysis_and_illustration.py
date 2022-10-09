#%%

from hashlib import new
import pandas as pd
import numpy as np

df = pd.read_json('Result_Pistachio.json')


# %%
import numpy as np
import matplotlib.pyplot as plt
import statistics

testnum = 7
treenum = 18
indexnum = 5
testdic = ['T1', 'T2', 'T3', 'T4','T5','T6','T7']

def idx_plot(idx_type):
    for i in range(testnum):
        data = df[df['spec_idx']==idx_type][df['test_number'] == testdic[i]]['pixel_array']
        newidx = data.index
        for j in range(treenum):
            arr = df[df['spec_idx']==idx_type][df['test_number'] == testdic[i]]['pixel_array'][newidx[j]]
            mean = statistics.mean(arr)
            average = sum(arr)/len(arr)
            plt.scatter(i,average)
    plt.show
    
ndvi = idx_plot('NDRE')


# %%
