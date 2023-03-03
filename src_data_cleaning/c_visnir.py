#%%#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from BaselineRemoval import BaselineRemoval

df_visnir = pd.read_json('../results/pistachio_visnir.json')

tdays = ['T5','T6','T7']
samdict = ['S1','S2','S3']
treenum = 18

#%%
##### plot all samples #####
def plt_nirvis(df):
    for k in range(len(tdays)):
        for i in range(treenum):
            plt.figure()
            plt.title("Test " + str(tdays[k]) + "   Tree " + str(i+1))
            for j in range(len(samdict)):
                test2=(df[df['test_number']==tdays[k]]\
                    [df['tree_id']==i+1][df['sample_number']==samdict[j]])
                yval = (test2['Reflect. %'])
                xval = (test2['Wvl'])           

                plt.plot(xval,yval)
                # plt.xticks([0 ,500 ,1000, 1500, 2000])  

plt_nirvis(df_visnir)

# %%
def bsl_rmv(df):
    for k in range(len(tdays)):
        for i in range(treenum):
            # fig, ax = plt.subplots(1,1)
            plt.figure()
            plt.title("Test " + str(tdays[k]) + "   Tree " + str(i+1))
            if k==0 and i==11-1:
                print('nothing happpens')
            else:
                for j in range(len(samdict)):
                    if k==0 and i==1-1 and j==1:
                        print('nothing happpens')
                    else:
                        test2=(df[df['test_number']==tdays[k]]\
                            [df['tree_id']==i+1][df['sample_number']==samdict[j]])
                        input_array = (test2['Reflect. %'])
                        xval = (test2['Wvl'])           
                        baseObj=BaselineRemoval(input_array)
                        Zhangfit_output=baseObj.ZhangFit()
                        plt.plot(xval,Zhangfit_output)
                    # plt.xticks([0 ,500 ,1000, 1500, 2000])         

bsl_rmv(df_visnir)

# %%
###### Average per day for each tree
def avg_pday(df):
    dfn = pd.DataFrame(columns=['test_number','tree_id','Wvl','Reflect. %'])
    dfmaster = pd.DataFrame(columns=['test_number','tree_id','Wvl','Reflect. %'])
    for k in range(len(tdays)):
        for i in range(treenum):
            avg = list([])
            for j in range(len(samdict)):
                if k==0 and i==1-1 and j==1:
                    print('nothing happens')
                else:    
                    val=(df[df['test_number']==tdays[k]]\
                        [df['tree_id']==i+1][df['sample_number']==samdict[j]])
                    yval = (val['Reflect. %']).values
                
                avg.append(yval)
            avg = sum(avg)/len(avg)
            xval = (val['Wvl']).values
            dfn['Wvl']=xval
            dfn['Reflect. %']=avg
            dfn['test_number']=tdays[k]
            dfn['tree_id']= i+1

            dfmaster = pd.concat([dfmaster,dfn],ignore_index=True)
            # plt.plot(xval,yval)

    return dfmaster

df_visnir_avg_c = avg_pday(df_visnir)

# %%
#### Check avg values are correct#####
def plt_chk_nirvis(df,dfc):
    for k in range(len(tdays)):
        for i in range(treenum):
            plt.figure()
            plt.title("Test " + str(tdays[k]) + "   Tree " + str(i+1))
            testc=(dfc[dfc['test_number']==tdays[k]]\
                    [dfc['tree_id']==i+1])
            yval = (testc['Reflect. %'])
            xval = (testc['Wvl'])   
            plt.plot(xval,yval,'g')
            for j in range(len(samdict)):
                test2=(df[df['test_number']==tdays[k]]\
                    [df['tree_id']==i+1][df['sample_number']==samdict[j]])
                yval = (test2['Reflect. %'])
                xval = (test2['Wvl'])           

                plt.plot(xval,yval,'r')
                # plt.xticks([0 ,500 ,1000, 1500, 2000])  

plt_chk_nirvis(df_visnir,df_visnir_avg_c)

# %%
### Baseline removal pday
def bsl_rmv(df):
    for k in range(len(tdays)):
        for i in range(treenum):
            # fig, ax = plt.subplots(1,1)
            plt.figure()
            plt.title("Test " + str(tdays[k]) + "   Tree " + str(i+1))
            
            test2=(df[df['test_number']==tdays[k]]\
                [df['tree_id']==i+1])
            input_array = (test2['Reflect. %'])
            xval = (test2['Wvl'])           
            baseObj=BaselineRemoval(input_array)
            Zhangfit_output=baseObj.ZhangFit()
            plt.plot(xval,Zhangfit_output)
                # plt.xticks([0 ,500 ,1000, 1500, 2000])         
bsl_rmv(df_visnir_avg_c)

#%%
df_visnir_avg_c.to_json('../results_cleaned/pistachio_visnir_c.json')

# %%
