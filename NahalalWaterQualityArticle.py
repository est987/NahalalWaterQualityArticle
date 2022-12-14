import os
import pandas as pd,numpy as np #geopandas as gpd, 


# import the fun to crete the plot

### connect to local data - removed
# fp='//fs01/users/matanb/GitHub/NahalalWaterQualityArticle'
# os.chdir(fp)

os.getcwd()

from def_plot import *
from helperMethods import *

################   get the data  #################

folder ='data'

# folder ='//fs01/users/matanb/GitHub/NahalalWaterQualityArticle/data'
df = pd.read_csv(os.path.join(folder,'final_df.csv'), parse_dates=['sample_date'])
point_measurement_data= pd.read_csv(os.path.join(folder,'point_measurement_data.csv'))
rain_data=pd.read_excel(os.path.join(folder,'rain_data.xlsx'), sheet_name='rain')
#sample_date_seasonrain=pd.read_excel(os.path.join(folder,'sample_date.xlsx'), sheet_name='sample_date')
#%%


###############   delect data #########################
# drop TON, IC
df_drop=df.drop(['TON','IC'], axis=1)
# drop 10/12/19
df_drop=df_drop.loc[df['sample_date']!='2019-11-12 00:00:00',:]
#select  ids 1-13
df_13=df_drop.loc[df_drop['id']<=13,:]
# drop id 6
df_13=df_13.loc[df_13['id']!=6,:]


#####################  variables to plots ######################
# pol list
poll1 = ['pH', 'EC','TOC']
poll2 =['Na','Cl']
poll3 = ['N-NH4', 'N-NO3', 'N-NO2','TN']
poll4=['P-PO4','K','Ca','Mg','S-SO4']
poll5 = ['Mn', 'Zn','Cu', 'Fe','Mo','B','Al','Co']
grop_poll_list=[poll1]+[poll2]+[poll3]+[poll4]+[poll5]
poll_list=poll1+poll2+poll3+poll4+poll5
poll_list_unite=[i+' mg/l' for i in poll_list]
rain_columns=['Cumulative_ rain','day_2','day_5','spi_2','spi_5']
tick_dates = pd.Series(df['sample_date'].unique()).sort_values()
id_groups=[np.array([1, 2, 3, 4, 5, 6, 7]),
                   np.array([8,9, 10, 11, 12, 13])]

poll_std = {'TN':10,
            'TP':1,
            'Cl':400,
            'Na':200,
            'Hg':0.0005,
            'Cr':0.05,
            'Ni':0.05,
            'Pb':0.008,
            'Cd':0.005,
            'Zn':0.2,
            'As':0.1,
            'Cu':0.02}





###########################  fun data perparation  #######################
#%%
#################### add rain data ##############################   
tick_dates,poll_data_join_rain=add_columns_sum_rain_between_dates(df_13,rain_data)
#tick_dates=add_columns_sum_rain_between_dates(df_13,rain_data)
#%%

####### TransformationData  to mean and st##############
dict_Transformation_df=TransformationData_fun(df_13,poll_list)
Standard_norm_data_13=dict_Transformation_df['Standard']

#%%
# grop data by id and spi index 
poll_data_join_rain_celect_columns=poll_data_join_rain.loc[:,['sample_date','id','spi_grop']+poll_list]
poll_data_join_rain_celect_columns['spi_grop']=poll_data_join_rain_celect_columns['spi_grop'].fillna('other')
ID_array=poll_data_join_rain_celect_columns['id'].values
id_grop= ['Upstream' if x in [1,2,3] else 'Middlestream' if x in [4,5,7,8,9] else 'Downstream' for x in ID_array]
poll_data_join_rain_celect_columns.insert(1, 'id_grop',id_grop)

grop_3=[str(a)+'_'+str(b) for a,b in zip(poll_data_join_rain_celect_columns['spi_grop'],poll_data_join_rain_celect_columns['id_grop'])]
poll_data_join_rain_celect_columns.insert(3, 'grop',grop_3)



### RUN PLOT GRAPHS

###
setStyle()

#save plot folder 
savefolder = '//fs01/users/matanb/GitHub/NahalalWaterQualityArticle/graphs/31102022'
#%%
#1
a,b,c,d=grop_data_mean_id(Standard_norm_data_13, poll_list)
heatmap_meangrop_id(d,savefolder)
#%%
#2 
distance_elevation(point_measurement_data,savefolder)
#%%
#3.1
Cumulative_rain_plot(tick_dates,rain_data,savefolder)
#3.2 3.3
spi_plot(tick_dates,savefolder)

#3.4
rgesin_spi(tick_dates,savefolder)
#%%
#4
a=grop_plot(poll_data_join_rain_celect_columns,poll_list,grop_poll_list,savefolder)
#%%
#5.1
duration_curve_1(poll_data_join_rain_celect_columns,poll_list,grop_poll_list,poll_std,savefolder)
#%%
#5.2
B=duration_curve_2(poll_data_join_rain_celect_columns,poll_list,grop_poll_list,poll_std,savefolder)











