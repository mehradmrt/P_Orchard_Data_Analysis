# %%
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

######## Constants ########
A_rtdr = 'C:\\Users\\Students\\Box\\Research' \
            + '\\IoT4ag\\Project_ Water Stress' \
                + '\\Data Collection\\Almond\\Ground Data' \
                    + '\\Almond_sap_weather\\Almond_data'
A_sensor_num = 6

P_rtdr = 'C:\\Users\\Students\\Box\\Research' \
            + '\\IoT4ag\\Project_ Water Stress\\Data Collection' \
                + '\\Pistachio\\Ground Data\\Pistachio_sap_weather'
P_sensor_num = 6

Almond_coef = SAP_SENSOR_COEFFICIENTS = [
    {"a": 1, "b": 1}, #Sensor 1
    {"a": 1, "b": 1}, #Sensor 2
    {"a": 1, "b": 1}, #Sensor 3
    {"a": 1, "b": 1}, #Sensor 4
    {"a": 1, "b": 1}, #Sensor 5
    {"a": 1, "b": 1} #Sensor 6
]


######## Class #########

class file_handle:
    def __init__(self, root, sensor_num):
        self.dir = root     # Root directory to the raw data
        self.num = sensor_num        # Number of sap sensors
        self.coef = Almond_coef
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

    def sap_correct(self):
        newsap = self.file_mixer_sap()
        for i in range(self.num):
            id = 'TREW ' + str(i+1)
            temp = newsap[newsap["Sensor ID"] == id]

            V1 = (temp['Value 1'])
            dT = (V1 - 1000)/20
            dTmin = min(dT)
            K = (dT-dTmin)/dT
            nu = 118.99 * 10**-6 * K**(1.231)
            # print(nu)
            
            
            V2 = (temp['Value 2'])
            

            temp['Value 1'] = nu
            temp['Value 2'] = 

        return newsap

    


    # def time_localize(self):
    #     sap_data = self.file_mixer_sap()
    #     sap_data['Date and Time'] = pd.to_datetime(sap_data['Date and Time']) \
    #                          .dt.tz_localize('UTC') \
    #                          .dt.tz_convert('America/LOS_ANGELES')

    #     weather_data = self.file_mixer_weather()                     
    #     weather_data['Date and Time'] = pd.to_datetime(weather_data['Date and Time']) \
    #                          .dt.tz_localize('UTC') \
    #                          .dt.tz_convert('America/LOS_ANGELES')

    #     return sap_data,weather_data



#### Calling Class ####

# Almond = file_handle(A_rtdr,A_sensor_num)
# A_sap = Almond.file_mixer_sap()
# A_weather = Almond.file_mixer_weather()

# Pistachio = file_handle(P_rtdr,P_sensor_num)
# P_sap = Pistachio.file_mixer_sap()
# P_weather = Pistachio.file_mixer_weather()



# %%
Almond = file_handle(A_rtdr,A_sensor_num)
sap_data = Almond.sap_correct()
weather_data = Almond.file_mixer_weather()

#####################
# ax = plt()

newsap = sap_data[(sap_data["Sensor ID"] == 'TREW 6')]
# newsap = sap_data[(sap_data["Sensor ID"] == 'TREW 6') & (sap_data["Value 2"]<2500)]
# newsap = newsap[1000:2000]
# newwed = weather_data[1000:2000]
# newsap['Value 1'] = (newsap['Value 1']-newsap['Value 1'].min())*300/newsap['Value 1'].max()

newsap.plot(kind = 'scatter',
        x = 'Date and Time',
        y = 'Value 1',
        color = 'blue')

# newwed.plot(kind = 'scatter',
#         x = 'Date and Time',
#         y = 'Temperature [℃]',
#         color = 'red')

plt.title('Sap Flow')
# plt.autoscale(enable=True, axis='y', tight=True)
plt.show()



############################ Visualization #################################
# ax = plt.twinx()

# newsap = sap_data[(sap_data["Sensor ID"] == 'TREW 3')]
# newsap = newsap[1000:2000]
# newwed = weather_data[1000:2000]
# newsap['Value 1'] = (newsap['Value 1']-newsap['Value 1'].min())*300/newsap['Value 1'].max()

# newsap.plot(kind = 'scatter',
#         x = 'Date and Time',
#         y = 'Value 1',
#         color = 'blue',ax=ax)

# newwed.plot(kind = 'scatter',
#         x = 'Date and Time',
#         y = 'Temperature [℃]',
#         color = 'red',ax=ax)

# plt.title('Sap Flow')
# plt.autoscale(enable=True, axis='y', tight=True)
# plt.show()



# %%
