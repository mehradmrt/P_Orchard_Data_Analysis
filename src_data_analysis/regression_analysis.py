#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#%%
###### SWP and Leaf Temperature linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x = np.reshape(np.array(df_lt['leaf_temp']),(-1,1))
y = np.array(df_swp['SWP'])

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .1)
x_train = x
y_train = y

lr = LinearRegression()
lr.fit(x_train,y_train)
# y_pred = lr.predict(x_train)
print('R^2 {}'.format(lr.coef_)) 

plt.scatter(x_train, y_train, color = "black")
plt.plot(x_train, lr.predict(x_train), color = "blue")
plt.title("Linear Regression")
plt.xlabel("Leaf Temp")
plt.ylabel("SWP")
plt.show()

#%%
###### SWP and Indexes Temperature linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

i = 2
imidx = df_im.loc[df_im['spec_idx'] == idxdic[i]]['median']
# x = np.reshape(np.array(imidx),(-1,1))
# y = np.array(df_swp['SWP'])
x = np.reshape(np.array(df_swp['SWP']),(-1,1))
y = np.array(imidx)
# x = np.reshape(np.array(df_lt['leaf_temp']),(-1,1))
# y = np.array(imidx)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .1)
x_train = x
y_train = y

lr = LinearRegression()
lr.fit(x_train,y_train)
# y_pred = lr.predict(x_train)
print('R^2: {}'.format(lr.coef_)) 

plt.scatter(x_train, y_train, color = "black")
plt.plot(x_train, lr.predict(x_train), color = "blue")
plt.title("Linear Regression")
# plt.xlabel("{}".format(idxdic[i]))
# plt.ylabel("SWP")
plt.show()

# %%

i = 6
swp_mn = np.array(df_swp.loc[df_swp['test_number'] == testdic[i]]['SWP'])
lt_mn = np.array(df_lt.loc[df_lt['test_number'] == testdic[i]]['leaf_temp'])

x = np.reshape(lt_mn,(-1,1))
y = np.array(swp_mn)
# x = np.reshape(np.array(df_swp['SWP']),(-1,1))
# y = np.array(imidx)
# x = np.reshape(np.array(df_lt['leaf_temp']),(-1,1))
# y = np.array(imidx)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .1)
x_train = x
y_train = y

lr = LinearRegression()
lr.fit(x_train,y_train)
# y_pred = lr.predict(x_train)
print('R^2: {}'.format(lr.coef_)) 

plt.scatter(x_train, y_train, color = "black")
plt.plot(x_train, lr.predict(x_train), color = "blue")
plt.title("Linear Regression")
# plt.xlabel("{}".format(idxdic[i]))
# plt.ylabel("SWP")
plt.show()
# %%

# %%

    # i,j = 0,1  #indexid,testid
for i in range(indexnum):
    for j in range(testnum):

        swp_mn = np.array(df_swp.loc[df_swp['test_number'] == testdic[j]]['SWP'])
        # lt_mn = np.array(df_lt.loc[df_lt['test_number'] == testdic[i]]['leaf_temp'])
        imidx = np.array(df_im.loc[df_im['spec_idx'] == idxdic[i]]\
            [df_im['test_number'] == testdic[j]]['median'])

        x = np.reshape(imidx,(-1,1))
        y = np.array(swp_mn)
        # x = np.reshape(np.array(df_swp['SWP']),(-1,1))
        # y = np.array(imidx)
        # x = np.reshape(np.array(df_lt['leaf_temp']),(-1,1))
        # y = np.array(imidx)

        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .1)
        x_train = x
        y_train = y

        lr = LinearRegression()
        lr.fit(x_train,y_train)
        # y_pred = lr.predict(x_train)
        print('R^2: {}, {} {}'.format(lr.coef_,idxdic[i],testdic[j])) 

        plt.scatter(x_train, y_train, color = "black")
        plt.plot(x_train, lr.predict(x_train), color = "blue")
        plt.title("Test {}".format(testdic[j]))
        plt.xlabel("{}".format(idxdic[i]))
        plt.ylabel("SWP")
        plt.show()

# %%
