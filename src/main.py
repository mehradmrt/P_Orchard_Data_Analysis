# %%
import pandas as pd
import os
import glob

##### Constants #######
rtdr = 'C:\\Users\\Students\\Box\\Research' \
            + '\\IoT4ag\\Project_ Water Stress' \
                + '\\Data Collection\\Almond\\Ground Data' \
                    + '\\Almond_sap_weather\\Almond_data'
                
sensor_num = 6

##### Class #####

class file_handle:
    def __init__(self, root, sensor_num):
        self.dir = root     # Root directory to the raw data
        self.num = sensor_num        # Number of sap sensors

    def file_mixer_sap(self):
        files = glob.glob(self.dir + '\\Data_TREWid' + '*.csv')
        pack = pd.concat(map(pd.read_csv, files), ignore_index=True)
        # print(pack)
        return pack

    def file_mixer_weather(self):
        files = glob.glob(self.dir + '\\Data_weather' + '*.csv')
        pack = pd.concat(map(pd.read_csv, files), ignore_index=True)
        # print(pack)
        return pack    


#### Calling Class ####

call = file_handle(rtdr,sensor_num)
sap_data = call.file_mixer_sap()
weather_data = call.file_mixer_weather()

# %%



# %%
