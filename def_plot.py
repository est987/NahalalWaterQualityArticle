import os 
import datetime
import pandas as pd,numpy as np #geopandas as gpd,
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from matplotlib import rcParams, ticker

####  organiz the data ###########
def add_columns_sum_rain_between_dates(poll_data,rain_data):
   tick_dates = pd.DataFrame(poll_data['sample_date'].unique(),columns=['sample_date']).sort_values(by='sample_date')
   tick_dates=tick_dates.assign(daydigom=tick_dates.index)
 
   for c in list(rain_data.columns)[0:]:
       add_columns=tick_dates['sample_date'].apply(lambda x:rain_data.loc[rain_data['date']==x,c].iloc[-1])
       tick_dates=tick_dates.assign(add_columns=add_columns)
       tick_dates=tick_dates.rename(columns={"add_columns":c})

   #4 group 
   g_list=list()
   for s2,s5 in zip(tick_dates['spi_2'],tick_dates['spi_5']):
      
       if (s2>=1) & (s5>=1):
           g='wet2wet5'
          
       elif   (s2<1) & (s5<1):   
           g='dry2dry5'
         
       elif (s2>=1) & (s5<1):
           g='wet2dry5'
          
       elif (s2<1) & (s5>=1):
           g='dry2wet5'
          
       else:
           g=np.nan
       g_list.append(g) 
   tick_dates=tick_dates.assign(spi_group= g_list)
   
   #nwe_hy  
   nwe_hy=[0]*34  
   nwe_hy[22]=1
   nwe_hy[33]=1
   tick_dates.insert(9,'nwe_hy', nwe_hy )

  
   # join rain data to poll_data
   poll_data_join_rain=poll_data.merge(tick_dates,left_on='sample_date', right_on='sample_date')
 
   return tick_dates,poll_data_join_rain



def TransformationData_fun(df,list_columns):
    x=df.loc[:,list_columns].values
    
    #StandardScaler
    Standard_Scaler = preprocessing.StandardScaler()
    Standard_transform = Standard_Scaler.fit_transform(x)
    norm_Standard = pd.DataFrame( Standard_transform,columns=list_columns,index=df.index) 
    df_norm_Standard=df[['id','sample_date']].merge(norm_Standard,left_index=True,right_index=True)
    
    #minmax
    minmax_scaler = preprocessing.MinMaxScaler()
    x_scaled_MinMax = minmax_scaler.fit_transform(x)
    df_norm_MinMax = pd.DataFrame(x_scaled_MinMax,columns=list_columns,index=df.index)
    df_norm_MinMax=df[['id','sample_date']].merge(df_norm_MinMax,left_index=True,right_index=True)
    
    dict_TransformationData={'not_norm':df,
                             'Standard': df_norm_Standard,
                             'minmax':df_norm_MinMax}
    return dict_TransformationData 


def group_data_mean_id(df, poll_list):
    df_melt=pd.melt(df,id_vars=['id','sample_date'], value_vars= poll_list)
    df_group =  df_melt.groupby(['id','variable']).mean().reset_index()
    df_pivot= df_group.pivot(index='variable', columns='id', values='value')
    df_pivot_reindex= df_pivot.reindex(poll_list)
    return df_melt,  df_group, df_pivot,df_pivot_reindex

def open_close_dam_data(data,open_close_date):
    pass

  
#  fun to plot 
def create_new_folder(newpath):
    if not os.path.isdir(newpath):
       os.makedirs(newpath)

def yLabel(p):
    if p == 'pH':
        return p
    elif p == 'EC':
        return 'EC (μS/cm)'
    else:
        return '{} (mg/L)'.format(p) # concentration

def heatmap_meangroup_id(meangroup_df,save_plot):
     cbar_kws={}
     annot_kws={"size":8}
     fig,ax= plt.subplots(figsize=(8,6))
     cmap = sns.color_palette("rocket_r", as_cmap=True)
     sns.heatmap(meangroup_df, annot=True,linewidths=.5,cbar_kws=cbar_kws,ax=ax, cmap=cmap, annot_kws=annot_kws)
     
     cbar = ax.collections[0].colorbar
     cbar.ax.set_ylim(-1.2,1.5)
     cbar.ax.set_yticks(list(np.arange(-1.1,1.6,0.5)))
     #cbar.ax.tick_params(labelsize=50)
     ax.tick_params(axis='x', labelrotation=0 ,which='major')#, labelsize=50)
     ax.tick_params(axis='y', labelrotation=0 ,which='major')#, labelsize=50)
     ax.tick_params(axis='both', which='minor')#, labelsize=50)
     ax.set_xlabel('Sample ID')#,size=50)
     ax.set_ylabel('Pollutant')#,size=50)
     
     
     ax.tick_params(axis='both', which='minor')#, labelsize=50)
     create_new_folder(os.path.join(save_plot,'heatmap_meam_id'))
     fig.savefig(os.path.join(save_plot,'heatmap_meam_id' ,'heatmap.png'), bbox_inches='tight', dpi=300)
  






################ plot by id ( Elevation and  distance_kishon) division to group################################
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        
      
        if  str(point['val'])=='12.0':
          
            ax.text(point['x']+50, point['y']-4, str(point['val']),fontsize=15)
        elif  str(point['val'])=='13.0':   
            ax.text(point['x']+50, point['y']+4, str(point['val']),fontsize=15)
            
        else:
            ax.text(point['x']+50, point['y']-1, str(point['val']),fontsize=15)

def distance_elevation(measurement_data,save_plot):
    measurement_data=measurement_data.sort_values(by='id')
    fig,ax= plt.subplots(figsize=(12,5))
    # 1 ccater plot
    sns.scatterplot(data=measurement_data, x="distance_longest_flow", y="elevation_1", hue="group_id",s=200,ax=ax)
    
    ax.set_xlabel('Distance from channel head (m)')#,size=40)
    ax.set_ylabel('Elevation (m)')#,size=40)
    ax.tick_params(axis='both', which='major')#, labelsize=40)
    ax.tick_params(axis='both', which='minor')#, labelsize=35)
    ax.legend(title="")#, fontsize=20, title_fontsize=15)
    label_point(measurement_data.distance_longest_flow, measurement_data.elevation_1, measurement_data.id, ax)  
    create_new_folder(os.path.join(save_plot,'point_id'))
    fig.savefig(os.path.join(save_plot,'point_id','distance_elevation.jpg'), bbox_inches='tight', dpi=300)
    
    # 2 line plot 
    # fig,ax= plt.subplots(figsize=(32,12))
    fig,ax= plt.subplots(figsize=(10,4))
    ax.plot(measurement_data['id'],measurement_data['distance_longest_flow'],'-k',linewidth=2.0)
    ax.plot(measurement_data['id'],measurement_data['distance_longest_flow'],'*k',markersize=9)
    ax.set_xlabel('Sample point ID')#,size=40)
    ax.set_ylabel('Distance from channel head (m)'),#,size=40)
    ax.tick_params(axis='both', which='major')#, labelsize=40)
    ax.tick_params(axis='both', which='minor') #, labelsize=35)

    ax_b=ax.twinx()

    #show right_spine
    ax_b.spines['right'].set_visible(True)

    ax_b.plot(measurement_data['id'],measurement_data['elevation_1'],'-r',linewidth=2.0)
    ax_b.plot(measurement_data['id'],measurement_data['elevation_1'],'*r',markersize=9)
    ax_b.set_xlabel('Sample point ID')#,size=40)
    ax_b.set_ylabel('Elevation (m)', color='red')#,size=40)
    ax_b.tick_params(axis='both', which='major', labelcolor='red')#, labelsize=40)
    ax_b.tick_params(axis='both', which='minor')#, labelsize=35)
    #ax.set_xticks(list(measurement_data['id']))
    ax.set_xticks(list(measurement_data['id']))     
    create_new_folder(os.path.join(save_plot,'point_id'))
    
    # axvline
    ax.axvline(3.5, color='grey', linewidth=2, linestyle='--')  
    ax.axvline(9.5, color='grey', linewidth=2, linestyle='--') 

    annotate_font_size=14
    annotate_font_color='blue'

    label='Upstream' 
    ax.annotate(label, # this is the text
    (1.5,7000), # this is the point to label
    textcoords="offset points", # how to position the text
    xytext=(0,0), # distance from text to points (x,y)
    ha='center',# horizontal alignment can be left, right or center 
    fontsize=annotate_font_size,
    color=annotate_font_color)
    label='Midstream'
    ax.annotate(label, # this is the text
    (6,7000), # this is the point to label
    textcoords="offset points", # how to position the text
    xytext=(0,0), # distance from text to points (x,y)
    ha='center',# horizontal alignment can be left, right or center 
    fontsize=annotate_font_size,
    color=annotate_font_color)
    label='Downstream'
    ax.annotate(label, # this is the text
    (11.5,7000), # this is the point to label
    textcoords="offset points", # how to position the text
    xytext=(0,0), # distance from text to points (x,y)
    ha='center',# horizontal alignment can be left, right or center 
    fontsize=annotate_font_size,
    color=annotate_font_color)
    
    fig.savefig(os.path.join(save_plot,'point_id','distance_elevation_line.jpg'), bbox_inches='tight', dpi=300)
    
         
# plot by sample_date (group by rain index)

# add_rain_data=tick_dates
# save_plot=savefolder
def Cumulative_rain_plot(add_rain_data,rain_data,save_plot): 
    # month_rain
    rain_data=rain_data.set_index('date',drop=False)
    rain_day3=rain_data.groupby(pd.Grouper(freq='D'))['rain_mm'].sum()
    rain_month3=rain_data.groupby(pd.Grouper(freq='M'))['rain_mm'].sum()

    ### set daterange
    # daterange = pd.date_range('2019-10-01','2022-01-01',freq='2M')
    daterange = pd.date_range('2019-10-01','2021-10-01',freq='2M')

    daterange=[x+datetime.timedelta(days=1) for x in daterange]
    create_new_folder(os.path.join(save_plot,'rain_plot'))
    nwe_hy_date=add_rain_data.loc[add_rain_data['nwe_hy']==1,'sample_date'].values
    nwe_hy=add_rain_data['HY'].unique()
    x_lavel_1=add_rain_data.iloc[11,:]['sample_date']
    x_lavel_2=add_rain_data.iloc[28,:]['sample_date']
    x_lavel_3=add_rain_data.iloc[33,:]['sample_date']

    final_date = '2021-10-01'
    ### FILTER UP to 3rd year
    add_rain_data = add_rain_data.loc[add_rain_data.sample_date<final_date]
    rain_month3 = rain_month3.loc[rain_month3.index <final_date]
    rain_day3=rain_day3.loc[rain_day3.index <final_date]
    def cumulativeRain1():
        #  Cumulative_ rain 1
        fig,ax= plt.subplots(figsize=(52,22))
        #plt.ylim(max(rain_month3), 0)
        ax2 = ax.twinx()
        ax2.invert_yaxis()
        ax2.bar(rain_month3.index,rain_month3, color ='blue', width = 4)
        ax2.set_ylabel('montthly rain mm',size=30)
        ax2.tick_params(axis='both', which='major', labelsize=40)
        ax2.tick_params(axis='both', which='minor', labelsize=35)
        ax.plot(add_rain_data['sample_date'],add_rain_data['Cumulative_ rain'],'-k',linewidth=2.0)
        ax.plot(add_rain_data['sample_date'],add_rain_data['Cumulative_ rain'],'*k',markersize=18)
        ax.set_xlim(add_rain_data.sample_date.min() - pd.DateOffset(5), add_rain_data.sample_date.max() + pd.DateOffset(5))
        ax.set_xticks(daterange)
        ax.axvline(pd.Timestamp('2020-10-01') , color='grey', linewidth=2, linestyle='--')
        ax.axvline(pd.Timestamp('2021-10-01'), color='grey', linewidth=2, linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.set_xlabel('Sample date',size=40)
        ax.set_ylabel('Cumulative rain mm',size=40)
        ax.tick_params(axis='both', which='major', labelsize=40)
        ax.tick_params(axis='both', which='minor', labelsize=35)
        label=nwe_hy[0]
        ax.annotate(label, # this is the text
        (x_lavel_1,200), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,0), # distance from text to points (x,y)
        ha='center',# horizontal alignment can be left, right or center
        fontsize=50,
        color='red')
        label=nwe_hy[1]
        ax.annotate(label, # this is the text
        (x_lavel_2,200), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,0), # distance from text to points (x,y)
        ha='center',# horizontal alignment can be left, right or center
        fontsize=50,
        color='red')
        label=nwe_hy[2]
        ax.annotate(label, # this is the text
        (x_lavel_3,200), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,0), # distance from text to points (x,y)
        ha='center',# horizontal alignment can be left, right or center
        fontsize=50,
        color='red')
        fig.savefig(os.path.join(save_plot,'rain_plot','Cumulative_rain_month_rain.jpg'), bbox_inches='tight')

    def cumulativeRain2():
        #  Cumulative_ rain 2
        fig,ax= plt.subplots(figsize=(52,22))
        #plt.ylim(max(rain_month3), 0)
        ax2 = ax.twinx()
        ax2.invert_yaxis()
        ax2.bar(rain_month3.index,rain_month3, color ='blue', width = 4)
        ax2.set_ylabel('montthly rain mm',size=30)
        ax2.tick_params(axis='both', which='major', labelsize=40)
        ax2.tick_params(axis='both', which='minor', labelsize=35)
        ax.plot(add_rain_data['sample_date'],add_rain_data['Cumulative_ rain_2'],'-k',linewidth=2.0)
        ax.plot(add_rain_data['sample_date'],add_rain_data['Cumulative_ rain_2'],'*k',markersize=18)
        ax.set_xlim(add_rain_data.sample_date.min() - pd.DateOffset(5), add_rain_data.sample_date.max() + pd.DateOffset(5))
        ax.set_xticks(daterange)
        ax.axvline(pd.Timestamp('2020-10-01'), color='grey', linewidth=2, linestyle='--')
        ax.axvline(pd.Timestamp('2021-10-01'), color='grey', linewidth=2, linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.set_xlabel('Sample date',size=40)
        ax.set_ylabel('Cumulative rain mm',size=40)
        ax.tick_params(axis='both', which='major', labelsize=40)
        ax.tick_params(axis='both', which='minor', labelsize=35)
        label= nwe_hy[0]
        ax.annotate(label, # this is the text
        (x_lavel_1,200), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,0), # distance from text to points (x,y)
        ha='center',# horizontal alignment can be left, right or center
        fontsize=50,
        color='red')
        label= nwe_hy[1]
        ax.annotate(label, # this is the text
        (x_lavel_2,200), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,0), # distance from text to points (x,y)
        ha='center',# horizontal alignment can be left, right or center
        fontsize=50,
        color='red')
        label=nwe_hy[2]
        ax.annotate(label, # this is the text
        (x_lavel_3,200), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,0), # distance from text to points (x,y)
        ha='center',# horizontal alignment can be left, right or center
        fontsize=50,
        color='red')
        fig.savefig(os.path.join(save_plot,'rain_plot','Cumulative_rain_month_rain_2.jpg'), bbox_inches='tight')

    def cumulativeRain3():
        #  Cumulative_ rain 3
        fig,ax= plt.subplots(figsize=(9,5))
        #plt.ylim(max(rain_month3), 0)
        ax2 = ax.twinx()

        ## add top and right border
        for pos in ['top','right']:
            ax2.spines[pos].set_visible(True)

        ax2.invert_yaxis()
        ax2.bar(rain_month3.index,rain_month3, color ='blue', width = 6, alpha=.7)
        ax2.set_ylabel('Monthly precipitation (mm)', color='blue')#,size=30)
        ax2.tick_params(axis='y', which='major', labelcolor='blue')#, labelsize=40)
        ax2.tick_params(axis='both', which='major')#, labelsize=40)
        ax2.tick_params(axis='both', which='minor')#, labelsize=35)
        grouped=add_rain_data.groupby(by='HY')
        for name,group in grouped:
         print(name)
         ax.plot(group['sample_date'],group['Cumulative_ rain_2'],'-k',linewidth=2.0)
         ax.plot(group['sample_date'],group['Cumulative_ rain_2'],'ok',markersize=3)
        ax.set_xlim(add_rain_data.sample_date.min() - pd.DateOffset(5), add_rain_data.sample_date.max() + pd.DateOffset(5))
        ax.set_xticks(daterange)
        ax.axvline(pd.Timestamp('2020-10-16'), color='grey', linewidth=2, linestyle='--')
        #ax.axvline(pd.Timestamp('2021-10-01'), color='grey', linewidth=2, linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.set_xlabel('Sample date')#,size=40)
        ax.set_ylabel('Cumulative precipitation (mm)')#,size=40)

        ax.tick_params(axis='both', which='major')#, labelsize=40)
        ax.tick_params(axis='both', which='minor')#, labelsize=35)
        label= nwe_hy[0]
        ax.annotate(label, # this is the text
        # (x_lavel_1,100), # this is the point to label
        (pd.Timestamp('2020-08-01'),100), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,0), # distance from text to points (x,y)
        ha='center',# horizontal alignment can be left, right or center
        fontsize=14,
        color='red')
        label= nwe_hy[1]
        ax.annotate(label, # this is the text
        # (x_lavel_2,100), # this is the point to label
        (pd.Timestamp('2021-08-01'),100), # this is the point to label

        textcoords="offset points", # how to position the text
        xytext=(0,0), # distance from text to points (x,y)
        ha='center',# horizontal alignment can be left, right or center
        fontsize=14,
        color='red')
        # label=nwe_hy[2]
        # ax.annotate(label, # this is the text
        # (x_lavel_3,200), # this is the point to label
        # textcoords="offset points", # how to position the text
        # xytext=(0,0), # distance from text to points (x,y)
        # ha='center',# horizontal alignment can be left, right or center
        # fontsize=14,
        # color='red')

        fig.savefig(os.path.join(save_plot,'rain_plot','Cumulative_rain_month_rain_3.jpg'), bbox_inches='tight', dpi=300)

    cumulativeRain3()
    
    def cumulativeRain_day():
        #  Cumulative_ rain 3
        fig,ax= plt.subplots(figsize=(9,5))
        #plt.ylim(max(rain_month3), 0)
        ax2 = ax.twinx()

        ## add top and right border
        for pos in ['top','right']:
            ax2.spines[pos].set_visible(True)

        ax2.invert_yaxis()
        ax2.bar(rain_day3.index,rain_day3, color ='blue', width = 6, alpha=.7)
        ax2.set_ylabel('Monthly precipitation (mm)', color='blue')#,size=30)
        ax2.tick_params(axis='y', which='major', labelcolor='blue')#, labelsize=40)
        ax2.tick_params(axis='both', which='major')#, labelsize=40)
        ax2.tick_params(axis='both', which='minor')#, labelsize=35)
        grouped=add_rain_data.groupby(by='HY')
        for name,group in grouped:
         print(name)
         ax.plot(group['sample_date'],group['Cumulative_ rain_2'],'-k',linewidth=2.0)
         ax.plot(group['sample_date'],group['Cumulative_ rain_2'],'ok',markersize=3)
        ax.set_xlim(add_rain_data.sample_date.min() - pd.DateOffset(5), add_rain_data.sample_date.max() + pd.DateOffset(5))
        ax.set_xticks(daterange)
        ax.axvline(pd.Timestamp('2020-10-16'), color='grey', linewidth=2, linestyle='--')
        #ax.axvline(pd.Timestamp('2021-10-01'), color='grey', linewidth=2, linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.set_xlabel('Sample date')#,size=40)
        ax.set_ylabel('Cumulative precipitation (mm)')#,size=40)

        ax.tick_params(axis='both', which='major')#, labelsize=40)
        ax.tick_params(axis='both', which='minor')#, labelsize=35)
        label= nwe_hy[0]
        ax.annotate(label, # this is the text
        # (x_lavel_1,100), # this is the point to label
        (pd.Timestamp('2020-08-01'),100), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,0), # distance from text to points (x,y)
        ha='center',# horizontal alignment can be left, right or center
        fontsize=14,
        color='red')
        label= nwe_hy[1]
        ax.annotate(label, # this is the text
        # (x_lavel_2,100), # this is the point to label
        (pd.Timestamp('2021-08-01'),100), # this is the point to label

        textcoords="offset points", # how to position the text
        xytext=(0,0), # distance from text to points (x,y)
        ha='center',# horizontal alignment can be left, right or center
        fontsize=14,
        color='red')
        # label=nwe_hy[2]
        # ax.annotate(label, # this is the text
        # (x_lavel_3,200), # this is the point to label
        # textcoords="offset points", # how to position the text
        # xytext=(0,0), # distance from text to points (x,y)
        # ha='center',# horizontal alignment can be left, right or center
        # fontsize=14,
        # color='red')

        fig.savefig(os.path.join(save_plot,'rain_plot','Cumulative_rain_day_rain_3.jpg'), bbox_inches='tight', dpi=300)

    cumulativeRain_day()
    
    
    

    
    
    
    
def  spi_plot(add_rain_data,save_plot): 
    create_new_folder(os.path.join(save_plot,'rain_plot'))
    daterange = pd.date_range('2019-10-01','2022-01-01',freq='2M')
    daterange=[x+datetime.timedelta(days=1) for x in daterange]
    fig,ax= plt.subplots(figsize=(52,22))
    ax_b=ax.twinx()
    ax.plot(add_rain_data['sample_date'],add_rain_data['day_2'],'-',color='tab:blue',linewidth=2.0)
    ax.plot(add_rain_data['sample_date'],add_rain_data['day_2'],'*',color='tab:blue',markersize=18)
    #ax.set_xlabel('id',size=60)
    #ax.set_ylabel('Elevation (m)',size=60)
    #ax.tick_params(axis='both', which='major', labelsize=50)
    #ax.tick_params(axis='both', which='minor', labelsize=45) 
    ax.set_ylabel('sum rain 2 day',color='tab:blue',fontsize=30)
    ax.tick_params(axis='y',labelcolor='tab:blue')
    ax.tick_params(axis='y', which='major', labelsize=40)
    ax.tick_params(axis='y', which='minor', labelsize=35) 
    
    ax_b.plot(add_rain_data['sample_date'],add_rain_data['day_5'],'-',color='tab:red',linewidth=2.0)
    ax_b.plot(add_rain_data['sample_date'],add_rain_data['day_5'],'*',color='tab:red',markersize=18)
    #ax.set_xlabel('id',size=60)
    #ax.set_ylabel('Elevation (m)',size=60)
    #ax.tick_params(axis='both', which='major', labelsize=50)
    #ax.tick_params(axis='both', which='minor', labelsize=45) 
    
    ax_b.set_ylabel('sum rain 5 day',color='tab:red',fontsize=30)
    ax_b.tick_params(axis='y',labelcolor='tab:red')
    ax_b.tick_params(axis='y', which='major', labelsize=40)
    ax_b.tick_params(axis='y', which='minor', labelsize=35) 
    
  
    ax.set_xlim(add_rain_data.sample_date.min() - pd.DateOffset(5), add_rain_data.sample_date.max() + pd.DateOffset(5))
    ax.set_xticks(daterange)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.set_xlabel('Sample date',size=30)
    ax.tick_params(axis='x', which='major', labelsize=40)
    ax.tick_params(axis='x', which='minor', labelsize=35) 
    fig.savefig(os.path.join(save_plot,'rain_plot','sum rain.jpg'), bbox_inches='tight')
    
    
  
    fig,ax= plt.subplots(figsize=(52,22))
    ax_b=ax.twinx()
    ax.plot(add_rain_data['sample_date'],add_rain_data['spi_2'],'-',color='tab:blue',linewidth=2.0)
    ax.plot(add_rain_data['sample_date'],add_rain_data['spi_2'],'*',color='tab:blue',markersize=18)
    ax.set_ylabel('spi 2',color='tab:blue',size=30)
    ax.tick_params(axis='y', which='major', labelsize=40)
    ax.tick_params(axis='y', which='minor', labelsize=35) 
    ax.tick_params(axis='y',labelcolor='tab:blue')
    
    ax_b.plot(add_rain_data['sample_date'],add_rain_data['spi_5'],'-',color='tab:red',linewidth=2.0)
    ax_b.plot(add_rain_data['sample_date'],add_rain_data['spi_5'],'*',color='tab:red',markersize=18)
    ax_b.set_ylabel('spi 5',color='tab:red',size=30)
    ax_b.tick_params(axis='y', which='major', labelsize=40)
    ax_b.tick_params(axis='y', which='minor', labelsize=35)
    ax_b.tick_params(axis='y',labelcolor='tab:red')
    
    ax.set_xlim(add_rain_data.sample_date.min() - pd.DateOffset(5), add_rain_data.sample_date.max() + pd.DateOffset(5))
    ax.set_xticks(daterange)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.set_xlabel('Sample date',size=30)
    
    ax.tick_params(axis='x', which='major', labelsize=40)
    ax.tick_params(axis='x', which='minor', labelsize=35) 
    
    fig.savefig(os.path.join(save_plot,'rain_plot','spi.jpg'), bbox_inches='tight')
   
 
   
def rgesin_spi(add_rain_data,save_plot):
    create_new_folder(os.path.join(save_plot,'rain_plot'))
    fig,ax= plt.subplots(figsize=(52,22))
    add_rain_data.plot.scatter(x='spi_2',y='spi_5',s=500,ax=ax)
    ax.axhline(y=1)
    ax.axvline(x=1)
    ax.set_xlabel('spi 2',size=30)
    ax.set_ylabel('spi 5',size=30)
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=35) 
    fig.savefig(os.path.join(save_plot,'rain_plot','rgesin_spi.jpg'), bbox_inches='tight')
    
    fig,ax= plt.subplots(figsize=(52,22))
    add_rain_data.plot.scatter(x='log_spi_2',y='log_spi_5',s=500,ax=ax)
    ax.axhline(y=0)
    ax.axvline(x=0)
    ax.set_xlabel('log spi 2',size=30)
    ax.set_ylabel('log spi 5',size=30)
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=35) 
    fig.savefig(os.path.join(save_plot,'rain_plot','rgesin_log_spi.jpg'), bbox_inches='tight')
    
    
    
# line plot by group




def group_plot(group_data,poll_list,group_poll_list,save_plot):
    a=0
    for polls in group_poll_list: #lopp on group poll 
      a=a+1

      for poll in polls:
         create_new_folder(os.path.join(save_plot,'line_group','group_poll_'+str(a)))

         print(poll)
         data_plot=group_data.loc[:,['id','id_group','spi_group','group',poll]]
         group_mean_id=data_plot.groupby(by='id_group')[poll].mean().reset_index(drop=False)
         group_mean_spi=data_plot.groupby(by='spi_group')[poll].mean().reset_index(drop=False)
         #group_mean_1=data_plot.groupby(by='group')[poll].mean()
         # group_mean=data_plot.groupby(by=['id_group','spi_group'])[poll].mean().reset_index(drop=False)
         group_mean=data_plot.groupby(by=['id_group','spi_group'])[poll].agg(['mean','sem']).reset_index(drop=False)

         group_mean= group_mean.loc[ group_mean['spi_group']!='wet2dry5',:]

        ### SET CATEGORICAL TYPES
         group_mean['id_group'] = ['Midstream' if x == 'Middlestream' else x for x in group_mean['id_group']]
         cat_type = CategoricalDtype(categories=["Upstream", "Midstream", "Downstream"], ordered=True)
         group_mean['id_group'] = group_mean['id_group'].astype(cat_type)

         cat_type = CategoricalDtype(categories=["dry2dry5", "dry2wet5", "wet2wet5",'other'], ordered=True)
         group_mean['spi_group'] = group_mean['spi_group'].astype(cat_type)
         
         # line plot
         palette=['yellow','orange','blue','red']

         fig,ax= plt.subplots(figsize=(5,4)) #"Set2"
         #sns.lineplot(x='id_group', y=poll,hue="spi_group", marker='o', data=group_mean, palette=palette, estimator=None,linewidth=1.5,ax=ax)
         sns.lineplot(x='id_group', y='mean', hue="spi_group", marker='o', data=group_mean, palette=palette, estimator=None,linewidth=1.5,ax=ax)
         ax.set_xlabel('Stream section')#,size=50)
         yLabel_unit=yLabel(poll)
         ax.set_ylabel(yLabel_unit)#,size=50)
         ax.tick_params(axis='both', which='major')#, labelsize=40)
         ax.tick_params(axis='both', which='minor')#, labelsize=35)
         ax.legend(title="")#, fontsize=40, title_fontsize=15)
         fig.savefig(os.path.join(save_plot,'line_group','group_poll_'+str(a),poll+'.jpg'), bbox_inches='tight',dpi=300)



    return data_plot,group_mean_id, group_mean_spi, group_mean






# duration_curve (standard)
def duration_curve_1(df,poll_list,group_poll_list,poll_std,save_plot):
   
    a=0
    for polls in group_poll_list: #lopp on group poll 
      a=a+1
      for poll in polls:
      
         create_new_folder(os.path.join(save_plot,'duration_curve','group_poll_'+str(a))) 
        
         data_poll=df.loc[:,['id','id_group','spi_group','group',poll]].reset_index(drop=True)  
         grouped=data_poll.groupby(by='id_group') 
         
         
         for name,group in grouped:
           fig,ax= plt.subplots(figsize=(5,4))
           try:
             std = poll_std[poll]
             ax.axhline(std, color='grey', linewidth=2, linestyle='--', label='{} ({} mg/l)'.format(poll, std))
             han, lab = ax.get_legend_handles_labels()
             ax.legend(labels=[lab[-1]], handles=[han[-1]], loc='best', frameon=False)
           except:
                 pass
             
           for i in group['id'].unique():
            print('ID',name,i)   
            data_plot=group.loc[group['id']==i,:]#.sort_values(by='sample_date').dropna().reset_index(drop=True) 
            data_plot=data_plot.sort_values(by=poll,ascending=False).reset_index(drop=True)
            data_plot= data_plot.assign(n=range(1, len(data_plot) + 1))
            data_plot=data_plot.assign(par=100* (data_plot['n']/max(data_plot['n'])))
            data_plot=data_plot.assign(parpoll=100* (data_plot[poll]/max(data_plot[poll])))
            ax.plot(data_plot['par'],data_plot[poll],linewidth=2.0, marker='o', fillstyle='none', alpha=0.7, label=str(i))
         
           
           ax.set_xlabel('percent of samples')#,size=25)
           yLabel_unit=yLabel(poll)
           ax.set_ylabel( yLabel_unit)#, size=25)
           ax.tick_params(axis='both', which='major')
           ax.tick_params(axis='both', which='minor')
    
           ax.legend(title='', ncol=1, loc='best', fontsize=10, frameon=False)#bbox_to_anchor=(1.0, 0.5)
           
           fig.savefig(os.path.join(save_plot,'duration_curve','group_poll_'+str(a),poll+'_'+name+'.jpg'), bbox_inches='tight', dpi=300)
     
      
   
def duration_curve_2(df,poll_list,group_poll_list,poll_std,save_plot):
    a=0
    for polls in group_poll_list: #lopp on group poll 
      a=a+1
      for poll in polls:
      
         create_new_folder(os.path.join(save_plot,'duration_curve_2','group_poll_'+str(a))) 
        
         data_poll=df.loc[:,['sample_date','id','id_group','spi_group','group',poll]].reset_index(drop=True)  
         grouped=data_poll.groupby(by='id_group') 
         fig,ax= plt.subplots(figsize=(5,4))
         try:
           std = poll_std[poll]
           ax.axhline(std, color='grey', linewidth=2, linestyle='--', label='{} ({} mg/l)'.format(poll, std))
           han, lab = ax.get_legend_handles_labels()
           ax.legend(labels=[lab[-1]], handles=[han[-1]], loc='best', frameon=False)
         except:
               pass
         for name,group in grouped:
            data_plot=pd.DataFrame(group.groupby(by='sample_date').mean())
            data_plot=data_plot.sort_values(by=poll,ascending=False).reset_index(drop=True)
            data_plot= data_plot.assign(n=range(1, len(data_plot) + 1))
            data_plot=data_plot.assign(par=100* (data_plot['n']/max(data_plot['n'])))
            ax.plot(data_plot['par'],data_plot[poll],linewidth=2.0, marker='o', fillstyle='none', alpha=0.7, label=name)
         ax.set_xlabel('Percent of samples (%)')
         yLabel_unit=yLabel(poll)
         ax.set_ylabel( yLabel_unit)
         ax.tick_params(axis='both', which='major')
         ax.tick_params(axis='both', which='minor')
         ax.legend(title='', ncol=1, loc='best', frameon=True, fontsize=11) # bbox_to_anchor=(1, 0.5),
         fig.savefig(os.path.join(save_plot,'duration_curve_2','group_poll_'+str(a),poll+'_'+name+'.jpg'), bbox_inches='tight', dpi=300)
   
  
             
    return group,data_plot





