
#%% Vis-NIR Extract
import pandas as pd
import glob
import os

tdays = ['T5','T6','T7']
samdict = ['S1','S2','S3']
nirdict = {'T5': '07_26_22', 'T6': '08_02_22', 'T7': '08_12_22'}
Exceptions = {'T5':['00','02'],'T6':['43'],'T7':[]}
vnir_dir = 'C:/Users/Students/Box/IoT-4Ag -Data/VIS_NIR/Pistachio_'
treenum = 18

def file_rename():
    for i in range(len(nirdict)):
        root  = vnir_dir + nirdict[tdays[i]]
        files = glob.glob(root+'/*.sed')
        for filename in files:
            base = os.path.splitext(filename)[0]
            newfiles = os.rename(filename, base + '.csv')
# ext_create = file_rename()        


def vis_nir():
    mdf = pd.DataFrame()
    df = pd.DataFrame()
    for i in range(len(nirdict)):
        root  = vnir_dir + nirdict[tdays[i]] 
        files = glob.glob(root+'/*.csv')

        tree_idx = 1
        sampnum = 0
        counter=0
        for filename in files:
            check=True
            if sampnum==3:
                tree_idx+=1
                sampnum=0

            if i==0 and counter==0:
                sampnum -= 1
                check=False
            elif i==0 and counter==2:
                sampnum+=1
                counter+=1
            elif i==0 and counter==31:
                tree_idx +=1
            elif i==1 and counter==43:
                check=False
                sampnum-=1

            if check==True:
                
                df = pd.read_csv(filename, skiprows=32, delimiter='\t')
                df.insert(0,"sample_number",samdict[sampnum])
                df.insert(0,"tree_id",tree_idx)
                df.insert(0,"test_number",tdays[i])
                # print(filename)
                # print(i,sampnum,tree_idx,counter)
                # print(df)
                
                mdf = pd.concat([mdf,df])
            
            sampnum += 1
            counter += 1
    
    return mdf
master = vis_nir()

#%% Raman Extract

ramdict = {'T5': '07_26_22', 'T6': '08_02_22', 'T7': '08_12_22'}

T5correct = ['B11L34','B11L33','B11L32','B8L6','B8L7','B8L8','B5L14',\
            'B5L15','B5L16','B2L30','B2L31','B2L32','A6R1','A6R2','A6R3','A3R8','A3R9','A3R10']

# %%
