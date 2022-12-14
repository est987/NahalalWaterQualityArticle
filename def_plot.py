import os 
import datetime
import pandas as pd,numpy as np #geopandas as gpd, 
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
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

   #4 grop 
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
   tick_dates=tick_dates.assign(spi_grop= g_list)
   
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


def grop_data_mean_id(df, poll_list):
    df_melt=pd.melt(df,id_vars=['id','sample_date'], value_vars= poll_list)
    df_grop =  df_melt.groupby(['id','variable']).mean().reset_index()
    df_pivot= df_grop.pivot(index='variable', columns='id', values='value')
    df_pivot_reindex= df_pivot.reindex(poll_list)
    return df_melt,  df_grop, df_pivot,df_pivot_reindex


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

def heatmap_meangrop_id(meangrop_df,save_plot):
     cbar_kws={}
     annot_kws={"size":8}
     fig,ax= plt.subplots(figsize=(8,6))
     cmap = sns.color_palette("rocket_r", as_cmap=True)
     sns.heatmap(meangrop_df, annot=True,linewidths=.5,cbar_kws=cbar_kws,ax=ax, cmap=cmap, annot_kws=annot_kws)
     
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
  






################ plot by id ( Elevation and  distance_kishon) division to grop################################
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
    fig,ax= plt.subplots(figsize=(32,12))
    # 1 ccater plot
    sns.scatterplot(data=measurement_data, x="distance_longest_flow", y="elevation_1", hue="grop_id",s=200,ax=ax)
    
    ax.set_xlabel('Distance longest flow (m)',size=40)
    ax.set_ylabel('Elevation (m)',size=40)
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=35) 
    ax.legend(title="", fontsize=20, title_fontsize=15)
    label_point(measurement_data.distance_longest_flow, measurement_data.elevation_1, measurement_data.id, ax)  
    create_new_folder(os.path.join(save_plot,'point_id'))
    fig.savefig(os.path.join(save_plot,'point_id','distance_elevation.jpg'), bbox_inches='tight')
    
    # 2 line plot 
    fig,ax= plt.subplots(figsize=(32,12))
    ax.plot(measurement_data['id'],measurement_data['distance_longest_flow'],'-k',linewidth=2.0)
    ax.plot(measurement_data['id'],measurement_data['distance_longest_flow'],'*k',markersize=18)
    ax.set_xlabel('Id',size=40)
    ax.set_ylabel('Distance longest flow (m)',size=40)
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.tick_params(axis='both', which='minor', labelsize=35) 
    ax_b=ax.twinx()
    ax_b.plot(measurement_data['id'],measurement_data['elevation_1'],'-r',linewidth=2.0)
    ax_b.plot(measurement_data['id'],measurement_data['elevation_1'],'*r',markersize=18)
    ax_b.set_xlabel('Id',size=40)
    ax_b.set_ylabel('Elevation (m)',size=40)
    ax_b.tick_params(axis='both', which='major', labelsize=40)
    ax_b.tick_params(axis='both', which='minor', labelsize=35) 
    #ax.set_xticks(list(measurement_data['id']))
    ax.set_xticks(list(measurement_data['id']))     
    create_new_folder(os.path.join(save_plot,'point_id'))
    
    # axvline
    ax.axvline(3.5, color='grey', linewidth=2, linestyle='--')  
    ax.axvline(9.5, color='grey', linewidth=2, linestyle='--') 

    label='Upstream' 
    ax.annotate(label, # this is the text
    (2,5000), # this is the point to label
    textcoords="offset points", # how to position the text
    xytext=(0,0), # distance from text to points (x,y)
    ha='center',# horizontal alignment can be left, right or center 
    fontsize=20,
    color='red') 
    label='Middlestream'
    ax.annotate(label, # this is the text
    (7,5000), # this is the point to label
    textcoords="offset points", # how to position the text
    xytext=(0,0), # distance from text to points (x,y)
    ha='center',# horizontal alignment can be left, right or center 
    fontsize=20,
    color='red') 
    label='Downstream'
    ax.annotate(label, # this is the text
    (12,5000), # this is the point to label
    textcoords="offset points", # how to position the text
    xytext=(0,0), # distance from text to points (x,y)
    ha='center',# horizontal alignment can be left, right or center 
    fontsize=20,
    color='red') 
    
    fig.savefig(os.path.join(save_plot,'point_id','distance_elevation_line.jpg'), bbox_inches='tight')
    
         
# plot by sample_date (grop by rain index) 
def Cumulative_rain_plot(add_rain_data,rain_data,save_plot): 
    # month_rain
    rain_data=rain_data.set_index('date',drop=False)
    rain_month3=rain_data.groupby(pd.Grouper(freq='M'))['rain_mm'].sum()
    daterange = pd.date_range('2019-10-01','2022-01-01',freq='2M')
    daterange=[x+datetime.timedelta(days=1) for x in daterange]
    create_new_folder(os.path.join(save_plot,'rain_plot'))
    nwe_hy_date=add_rain_data.loc[add_rain_data['nwe_hy']==1,'sample_date'].values
    nwe_hy=add_rain_data['HY'].unique()
    x_lavel_1=add_rain_data.iloc[11,:]['sample_date']
    x_lavel_2=add_rain_data.iloc[28,:]['sample_date']
    x_lavel_3=add_rain_data.iloc[33,:]['sample_date']

  
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
    
    #  Cumulative_ rain 3
    fig,ax= plt.subplots(figsize=(52,22))
    #plt.ylim(max(rain_month3), 0)
    ax2 = ax.twinx()
    ax2.invert_yaxis()
    ax2.bar(rain_month3.index,rain_month3, color ='blue', width = 4)
    ax2.set_ylabel('montthly rain mm',size=30)
    ax2.tick_params(axis='both', which='major', labelsize=40)
    ax2.tick_params(axis='both', which='minor', labelsize=35) 
    grouped=add_rain_data.groupby(by='HY')
    for name,group in grouped:
     print(name)
     ax.plot(group['sample_date'],group['Cumulative_ rain_2'],'-k',linewidth=2.0)
     ax.plot(group['sample_date'],group['Cumulative_ rain_2'],'*k',markersize=18)
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
    
    
    
    fig.savefig(os.path.join(save_plot,'rain_plot','Cumulative_rain_month_rain_3.jpg'), bbox_inches='tight')
    
    
    
    
    
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
    
    
    
# line plot by grop

def grop_plot(grop_data,poll_list,grop_poll_list,save_plot):
    a=0
    for polls in grop_poll_list: #lopp on grop poll 
      a=a+1
      for poll in polls:
         create_new_folder(os.path.join(save_plot,'line_grop','grop_poll_'+str(a))) 
         
         print( poll)
         data_plot=grop_data.loc[:,['id','id_grop','spi_grop','grop',poll]]
         grop_mean_id=data_plot.groupby(by='id_grop')[poll].mean().reset_index(drop=False)
         grop_mean_spi=data_plot.groupby(by='spi_grop')[poll].mean().reset_index(drop=False)
         #grop_mean_1=data_plot.groupby(by='grop')[poll].mean()
         grop_mean=data_plot.groupby(by=['id_grop','spi_grop'])[poll].mean().reset_index(drop=False)
         grop_mean= grop_mean.loc[ grop_mean['spi_grop']!='wet2dry5',:]
         
         # line plot  
         palette=['blue','orange','yellow','red']
         fig,ax= plt.subplots(figsize=(52,22)) #"Set2"
         sns.lineplot(x='id_grop', y=poll,hue="spi_grop", data=grop_mean,palette=palette, estimator=None,linewidth=15,ax=ax)
         ax.set_xlabel('id grop',size=50)
         yLabel_unit=yLabel(poll)
         ax.set_ylabel(yLabel_unit,size=50)
         ax.tick_params(axis='both', which='major', labelsize=40)
         ax.tick_params(axis='both', which='minor', labelsize=35) 
         ax.legend(title="", fontsize=40, title_fontsize=15)
         fig.savefig(os.path.join(save_plot,'line_grop','grop_poll_'+str(a),poll+'.jpg'), bbox_inches='tight')
   
        
         
    return data_plot,grop_mean_id, grop_mean_spi, grop_mean
         
    





# duration_curve (standard)
def duration_curve_1(df,poll_list,grop_poll_list,poll_std,save_plot):
   
    a=0
    for polls in grop_poll_list: #lopp on grop poll 
      a=a+1
      for poll in polls:
      
         create_new_folder(os.path.join(save_plot,'duration_curve','grop_poll_'+str(a))) 
        
         data_poll=df.loc[:,['id','id_grop','spi_grop','grop',poll]].reset_index(drop=True)  
         grouped=data_poll.groupby(by='id_grop') 
         
         
         for name,group in grouped:
           fig,ax= plt.subplots(figsize=(22,12))   
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
         
           
           ax.set_xlabel('percent of samples',size=25)
           yLabel_unit=yLabel(poll)
           ax.set_ylabel( yLabel_unit, size=25)
           ax.tick_params(axis='both', which='major', labelsize=40)
           ax.tick_params(axis='both', which='minor', labelsize=35) 
    
           ax.legend(title='Sample id', title_fontsize=25, ncol=1, loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=25, frameon=False)
           
           fig.savefig(os.path.join(save_plot,'duration_curve','grop_poll_'+str(a),poll+'_'+name+'.jpg'), bbox_inches='tight')
     
      
   
def duration_curve_2(df,poll_list,grop_poll_list,poll_std,save_plot):
    a=0
    for polls in grop_poll_list: #lopp on grop poll 
      a=a+1
      for poll in polls:
      
         create_new_folder(os.path.join(save_plot,'duration_curve_2','grop_poll_'+str(a))) 
        
         data_poll=df.loc[:,['sample_date','id','id_grop','spi_grop','grop',poll]].reset_index(drop=True)  
         grouped=data_poll.groupby(by='id_grop') 
         fig,ax= plt.subplots(figsize=(22,12))  
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
         ax.set_xlabel('percent of samples',size=25)
         yLabel_unit=yLabel(poll)
         ax.set_ylabel( yLabel_unit, size=25)
         ax.tick_params(axis='both', which='major', labelsize=40)
         ax.tick_params(axis='both', which='minor', labelsize=35) 
         ax.legend(title='Sample id', title_fontsize=25, ncol=1, loc='center left', bbox_to_anchor=(1.15, 0.5), fontsize=25, frameon=False)
         fig.savefig(os.path.join(save_plot,'duration_curve_2','grop_poll_'+str(a),poll+'_'+name+'.jpg'), bbox_inches='tight')
   
  
             
    return group,data_plot





