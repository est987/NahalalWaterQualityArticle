import pandas as pd
pd.set_option('display.max_columns', 50)
import matplotlib as mpl
mpl.use('Qt5Agg')
mpl.get_backend()
from scipy.stats import zscore


from def_plot import *
from helperMethods import setStyle
################   get the data  #################

folder ='data/'

### Load Data
df = pd.read_csv(f'{folder}final_df.csv', parse_dates=['sample_date'])
## set stream section
df['stream_section'] = ['Groundwater' if x == 6 else 'Upstream' if x in [1,2,3] else 'Midstream' if x in [4,5,7,8,9] else 'Downstream' for x in df['id']]



df_clean=df.loc[df['sample_date']!='2019-11-12 00:00:00',:]
df_clean = df_clean.loc[df_clean['id']<=13,:]
open_dam_date = pd.read_csv(f'{folder}open dam date.csv',parse_dates=['sample_date'])
point_measurement_data= pd.read_csv(f'{folder}point_measurement_data.csv')
rain_data=pd.read_excel(f'{folder}rain_data.xlsx', sheet_name='rain')


id_cols= ['id','stream_section','sample_date']
final_df= df_clean[id_cols]

poll_list = ['N-NO3', 'N-NO2','N-NH4','P-PO4','Cl','EC']

def returnCleanedTable(df, poll_list, id_cols):
    final_df = df[id_cols]

    for p in poll_list:
        print(p)
        df_p = df[id_cols+[p]].dropna()
        df_m = df_p.merge(zscore(df_p[[p]]).rename(columns={p:'z'}), left_index=True,right_index=True, how='outer')
        print(f'{len(df_m.loc[df_m.z >2.5])} samples of {len(df_p)} ({round(len(df_m.loc[df_m.z >3]) / len(df_p)*100,1) }%) removed for {p}:')
        print(df_m.loc[df_m.z >2.5])
        df_m.loc[df_m.z>2.5, p] = float('nan')
        final_df = final_df.merge(df_m[[p]], left_index=True,right_index=True, how='outer')
    return final_df

def add_rain_data(df, rain_data):
    ###add rain_data
    tick_dates, df_rain = add_columns_sum_rain_between_dates(df, rain_data)
    # tick_dates=add_columns_sum_rain_between_dates(df_13,rain_data)

    # group data by id and spi index 
    df_rain = df_rain.loc[:, ['sample_date', 'id', 'spi_group'] + poll_list].dropna()
    df_rain['spi_group'] = ['2-Dry;5-Dry' if x == 'dry2dry5' else '2-Dry;5-Wet' if x == 'dry2wet5' else '2-Wet;5-Wet'
                               for x in df_rain['spi_group']]
    # df_rain['spi_group']=df_rain['spi_group'].fillna('other')
    ID_array = df_rain['id'].values
    id_group = ['Upstream' if x in [1, 2, 3] else 'Midstream' if x in [4, 5, 7, 8, 9] else 'Downstream' for x in
                ID_array]
    df_rain.insert(1, 'stream_section', id_group)

    group_3 = [str(a) + '_' + str(b) for a, b in zip(df_rain['spi_group'], df_rain['stream_section'])]
    df_rain.insert(3, 'section_spi_group', group_3)

    return df_rain

final_df = returnCleanedTable(df_clean, poll_list, id_cols)

df_12=final_df.loc[df_clean['id']!=6,:]

df_12_rain = add_rain_data(df_12, rain_data)
df_12_rain.columns
df[poll_list].hist()
final_df.hist()
zscore(df['Cl'].dropna()).max()

