#%%

from sklearn.neural_network import MLPRegressor
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

testnum = 7
treenum = 18
indexnum = 5
testdic = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
idxdic = ['NDVI', 'GNDVI', 'OSAVI', 'LCI' ,'NDRE']
DOY = [158, 172, 186, 194, 207, 214, 224]

#%% Data preparation

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


#%%

mlp = MLPRegressor(hidden_layer_sizes=5,max_iter=2000)
mlp.fit(X_train, y_train)

y_predict = mlp.predict(X_test)
















































# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn import preprocessing
# from sklearn.utils import shuffle
# from sklearn import metrics
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.neural_network import MLPRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import ARDRegression
# from sklearn import ensemble
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.preprocessing import MinMaxScaler
# #%%
# n=0
# n_instances=5
# Angles=6
# xx=np.zeros((n_instances**Angles,Angles))
# for i in range(5):
#     for j in range(5):
#         for k in range(5):
#             for l in range(5):
#                 for p in range(5):
#                     for q in range(5):
#                         xx[n,:]=[i,j,k,l,p,q]
#                         n=n+1
# xx=(xx+1)*5
# xx= shuffle(xx, random_state=0)
# x_total=xx
# #xx = preprocessing.scale(xx)
# #%%
# dataset = pd.read_csv('results5by5.csv',header=0,
#                       skipinitialspace=True).values
# x = dataset[:,1:5].astype(float)

# #x = np.hstack((np.ones((x.shape[0],1)),x))
# y = dataset[:,6:7].astype(float)
# #x , y = shuffle(x, y , random_state=0) 
# x = preprocessing.scale(x)
# scaler = MinMaxScaler()
# scaler.fit(y)
# normalized_y = scaler.transform(y)
# #yy = scaler.inverse_transform(normalized_y)
# x_train, x_test, y_train, y_test = train_test_split(x,normalized_y, test_size=0.1, random_state=2)

# #%% Neural Network

# params = {'hidden_layer_sizes': 400, 'activation': 'relu', 'solver': 'adam',
#           'alpha': 0.001, 'batch_size': 'auto' ,'learning_rate': 'constant','learning_rate_init':0.01,
#           'power_t':0.5,'max_iter':1000,'shuffle':True, 'tol':0.0001,'momentum':0.9,'validation_fraction':0.1,'beta_1':0.9,'beta_2':0.999}

# MLP = MLPRegressor(**params)

# MLP.fit(x_train, y_train)
# predictionMLP=MLP.predict(x_test)
# errorMLP = mean_absolute_error(y_test, predictionMLP)
# AccuracyMLP=MLP.fit(x_train,y_train).score(x_test,y_test)

# pipe2 = Pipeline([
#     ('classifier', MLPRegressor())
# ])
# grid_param2=[{
#         "classifier": [MLPRegressor()],
#         "classifier__hidden_layer_sizes": [400],
#         "classifier__activation":['relu'],
#         "classifier__solver":['adam'],
#         "classifier__alpha":[0.001],
#         "classifier__batch_size":['auto'],
#         "classifier__learning_rate":['constant'],
#         "classifier__learning_rate_init":[0.01],
#         "classifier__power_t":[0.5],
#         "classifier__max_iter":[1000],
#         #"classifier__tol":0.0001,
#         "classifier__momentum":[0.9],
#         "classifier__validation_fraction":[0.1],
#         "classifier__beta_1":[0.9],
#         "classifier__beta_2":[0.999]
#         }]

# gridsearch2 = GridSearchCV(pipe2, grid_param2, cv=10, verbose=0,n_jobs=-1) # Fit grid search
# best_model2 = gridsearch2.fit(x_train,y_train)
# print(best_model2.best_estimator_)
# print("The mean accuracy of the model is:",best_model2.score(x_test,y_test))
# pd.DataFrame(MLP.loss_curve_).plot()
# #%% prediction for all angles
# predictionAngles=MLP.predict(xx)
# A=max(predictionAngles)
# max_index = np.argmax(predictionAngles, axis=0)
# print(x_total[max_index,:])
