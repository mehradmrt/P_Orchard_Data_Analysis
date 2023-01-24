
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
master_visnir = vis_nir()

#%% Raman Extract
import pandas as pd
import glob

tdays = ['T5','T6','T7']
ramdict = {'T5': '07_26_22', 'T6': '08_02_22', 'T7': '08_12_22'}
T5correct = ['B11L34','B11L33','B11L32','B8L6','B8L7','B8L8','B5L14',\
            'B5L15','B5L16','B2L30','B2L32','A6R1','A6R2','A6R3','A3R8','A3R9','A3R10'] #'B2L31' is missing
T6correct = []
T7correct = []
corr_dict = {'T5': T5correct, 'T6': T6correct, 'T7': T7correct}
samdict = ['S1','S2','S3']
ram_dir = 'C:/Users/Students/Box/IoT-4Ag -Data/Ramandata/Pistachio-Leaves-Ramandata-'
treenum = 18

def Raman():
    mdf = pd.DataFrame()
    df = pd.DataFrame()
    
    for i in range(len(ramdict)):
        root  = ram_dir + ramdict[tdays[i]] 
        
        if i==0:
            
            tree_idx = 1
            for filematch in corr_dict[tdays[i]]:
                filepack = glob.glob(root+ '/' + filematch + '*.csv')
                sampnum = 0
                # print(filepack)
                # indices = [enum for enum, s in enumerate(filename) if  in s]
                for filename in filepack:
                    df = pd.read_csv(filename, skiprows=105)
                    df.insert(0,"sample_number",samdict[sampnum])
                    df.insert(0,"tree_id",tree_idx)
                    df.insert(0,"test_number",tdays[i])
                    mdf = pd.concat([mdf,df])
                    
                    sampnum += 1
                    # print(filename)
                    # print(df.iloc[:,4:6])
                
                tree_idx += 1


        if i==1:
            files = glob.glob(root+ '/*.csv')
            tree_idx = 1
            sampnum = 0
            for j in range(len(files)):
                if j==0:
                    counter=57
                if j==3:
                    counter=54
                if j==6:
                    counter=51
                if j==9:
                    counter=24
                if j==36:
                    counter=5                
                if j==45:
                    counter=15
                check=True
                if sampnum==3:
                    tree_idx+=1
                    sampnum=0
                
                if counter==11:                   
                    counter+=1

                if check==True:
                    filename = glob.glob(root+ '/SP_'+ str(counter) +'.csv')
                    # print(filename)

                    df = pd.read_csv(filename[0], skiprows=105)
                    df.insert(0,"sample_number",samdict[sampnum])
                    df.insert(0,"tree_id",tree_idx)
                    df.insert(0,"test_number",tdays[i])
                    mdf = pd.concat([mdf,df])
                    sampnum +=1
                    counter +=1
                    # print(df.iloc[:,0:3])

        if i==2:
            files = glob.glob(root+ '/*.csv')
            tree_idx = 1
            sampnum = 0
            counter=1
            for j in range(len(files)):
                check=True
                if sampnum==3:
                    tree_idx+=1
                    sampnum=0

                if check==True:
                    filename = glob.glob(root+ '/SP_'+ str(counter) +'.csv')
                    # print(filename)

                    df = pd.read_csv(filename[0], skiprows=105)
                    df.insert(0,"sample_number",samdict[sampnum])
                    df.insert(0,"tree_id",tree_idx)
                    df.insert(0,"test_number",tdays[i])
                    mdf = pd.concat([mdf,df])
                    sampnum +=1
                    counter +=1
                    # print(df.iloc[:,0:3])
    return mdf

master_ram = Raman()


# %%
###### Save results in .json
master_visnir.to_json('pistachio_visnir.json')
master_ram.to_json('pistachio_raman.json')

