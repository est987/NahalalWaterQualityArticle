import seaborn as sns, pandas as pd, numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

data = final_df
open_close_date = open_dam_date
#poll_list

savefolder = 'graphs/final/'

standards = {'P-PO4':[0.1, 0.5],
             'N-NO3':[4,12],
             'N-NH4':[0.2,2],
             'EC':[2000,5000],
             'Cl':[400,float('nan')]}
def open_close_dam_data_boxplot_final(data, open_close_date, savefolder):
    # join open dem
    cols = ['id','sample_date','open_dam']
    polls = ['EC','Cl','N-NO3', 'N-NO2','N-NH4','P-PO4']

    data_m = data.merge(open_close_date, on='sample_date')
    data_m = data_m.loc[data_m['open_dam'] != 'no_data', :]
    data_filt = data_m.loc[data_m['id'].isin([8,10,11]), cols+polls]
    data_melt = pd.melt(data_filt, id_vars=cols, value_vars=polls,var_name='element', value_name='conc')

    p = sns.catplot(data=data_melt, kind='box', x='id', y='conc', col='element', height=4, hue='open_dam', sharey=False, col_wrap=2, showfliers=False)
    p.set_titles("{col_name}", weight='bold')
    p.set_axis_labels('Sample ID', 'Concentration (mg/L)', weight='bold')
    p._legend.remove()


    for ax in p.axes:
        title = ax.get_title()
        ax.tick_params(labelleft=True)

        ##update ylabel
        if title == 'EC':
            ax.set_ylabel('Electrical conductivity (mS/cm)', weight='bold')
        else:
            ax.set_ylabel('Concentration (mg/L)', weight='bold')
        try:
            std = standards[title]
            print(f'{title} has  standard: {std}')
            ax.axhline(std[0], ls='--', color='green')
            ax.axhline(std[1], ls='--', color='red')
        except:
            print(f'{title} has no standard')

    handles, labels = p.axes[0].get_legend_handles_labels()
    labels = ['Open dam' if x == 'open' else 'Closed dam' for x in labels]
    custom_lines = [Line2D([0], [0], color='green', ls='--'),
                    Line2D([0], [0], color='red', ls='--')]
    for line in custom_lines:
        handles.append(line)
    custom_labels = ['High quality\n[upper limit]','Low quality\n[lower limit]']
    for label in custom_labels:
        labels.append(label)

    #.legend(handles=handles, labels=labels, title='')
    p.add_legend(dict(zip(labels, handles)),title='')
    p.savefig(f'{savefolder}open_close_dam.jpeg', bbox_inches='tight', dpi=300)

df=final_df
def analysisByStreamSection(df, savefolder, poll_list):

    data_cols = ['sample_date', 'id', 'stream_section']
    data_poll = df.loc[:, data_cols+poll_list].reset_index(drop=True)
    data_melt = pd.melt(data_poll, id_vars=data_cols, value_vars=poll_list,var_name='element', value_name='conc').dropna().sort_values(by='conc',ascending=False)
    data_melt['perc'] = 0
    data_melt['n'] = 0

    # p='N-NH4'
    for p in poll_list:
        for iss in data_melt.stream_section.unique():
            data_melt.loc[(data_melt.element==p) & (data_melt.stream_section==iss), 'perc'] = data_melt.loc[(data_melt.element==p) & (data_melt.stream_section==iss), 'conc'].rank(pct=True) * 100
            data_melt.loc[(data_melt.element==p) & (data_melt.stream_section==iss), 'perc'] = 100 - data_melt.loc[(data_melt.element==p) & (data_melt.stream_section==iss), 'perc']
    #

    data_melt['perc'].max()


    # cmap = sns.color_palette(['#66c2a5','#fc8d62','#8da0cb','#e78ac3'])
    # cmap = sns.color_palette('muted', 6)
    # del cmap[2:4]
    cmap = sns.color_palette(["#7eb0d5", "#bd7ebe", "#ffb55a", "#ffee65"])

    p = sns.relplot(data=data_melt, kind='line', x='perc', y='conc', col='element', hue='stream_section', col_wrap=2, height=4, aspect=.8,
                    linewidth = 1.5, alpha=.75, facet_kws={'sharey': False, 'sharex': True}, palette=cmap)

    #set title, labels, and tics
    p.set_titles("{col_name}", weight='bold')
    p.set_axis_labels('% of samples above concentration', 'Concentration (mg/L)', weight='bold')
    p._legend.remove()

    title='Cl'
    ## add standards to graphs
    i=0
    for ax in p.axes.flat:
        title = ax.get_title()
        ax.set_xlim(-1,100)
        ax.set_xticks(np.arange(0,101,10))
        ax.set_xticks(np.arange(0,101,5), minor=True)
        ax.tick_params(labelleft=True)

        if title == 'EC':
            ax.set_ylabel('Electrical conductivity (mS/cm)', weight='bold')
        else:
            ax.set_ylabel('Concentration (mg/L)', weight='bold')

        try:
            std = standards[title]
            print(f'{title} has  standard: {std}')
            ax.axhline(std[0], ls='--', color='green')
            ax.axhline(std[1], ls='--', color='red')
        except:
            print(f'{title} has no standard')

    handles, labels = p.axes[0].get_legend_handles_labels()
    custom_lines = [Line2D([0], [0], color='green', ls='--'),
                    Line2D([0], [0], color='red', ls='--')]
    for line in custom_lines:
        handles.append(line)
    custom_labels = ['High quality\n[upper limit]', 'Low quality\n[lower limit]']
    for label in custom_labels:
        labels.append(label)

    # .legend(handles=handles, labels=labels, title='')
    p.add_legend(dict(zip(labels, handles)), title='')
    plt.subplots_adjust( wspace=0.25)

    p.savefig(f'{savefolder}concentration_by_stream_section.jpeg', bbox_inches='tight', dpi=300)

### By SPI
df=df_12_rain
def bySectionSPI(df, poll_list, savefolder):

    data_cols = ['sample_date', 'id', 'stream_section', 'spi_group']
    data_poll = df.loc[:, data_cols+poll_list].reset_index(drop=True)
    data_melt = pd.melt(data_poll, id_vars=data_cols, value_vars=poll_list,var_name='element', value_name='conc').dropna().sort_values(by='conc',ascending=False)

    cmap = sns.color_palette(["#ffb55a", "#bd7ebe", "#7eb0d5"])

    data_melt.groupby(['stream_section','spi_group','element']).mean()
    #p=sns.FacetGrid(data=data_melt, col='element', col_wrap=2,  height=4, aspect=.9, sharex=True, sharey=False)

    p = sns.relplot(data=data_melt, kind='line', x='stream_section', y='conc', marker='o', col='element', hue='spi_group', col_wrap=2,
                    height=4, aspect=.8,linewidth=1.5, alpha=.75, ci=66, facet_kws={'sharey': False, 'sharex': True}, palette=cmap)


    # set title, labels, and tics
    p.set_titles("{col_name}", weight='bold')
    p.set_axis_labels('Stream section', 'Concentration (mg/L)', weight='bold')
    p._legend.remove()

    title = 'Cl'
    ## add standards to graphs
    i = 0
    for ax in p.axes.flat:
        title = ax.get_title()
        ax.tick_params(labelleft=True)

        if title == 'EC':
            ax.set_ylabel('Electrical conductivity (mS/cm)', weight='bold')
        else:
            ax.set_ylabel('Concentration (mg/L)', weight='bold')

        try:
            std = standards[title]
            print(f'{title} has  standard: {std}')
            ax.axhline(std[0], ls='--', color='green')
            ax.axhline(std[1], ls='--', color='red')
        except:
            print(f'{title} has no standard')

    handles, labels = p.axes[0].get_legend_handles_labels()
    custom_lines = [Line2D([0], [0], color='green', ls='--'),
                    Line2D([0], [0], color='red', ls='--')]
    for line in custom_lines:
        handles.append(line)
    custom_labels = ['High quality\n[upper limit]', 'Low quality\n[lower limit]']
    for label in custom_labels:
        labels.append(label)

    # .legend(handles=handles, labels=labels, title='')
    p.add_legend(dict(zip(labels, handles)), title='')
    plt.subplots_adjust(wspace=0.2)

    p.savefig(f'{savefolder}concentration_by_spi_stream_section.jpeg', bbox_inches='tight', dpi=300)







data_poll = df.loc[df.id.isin([1,3]), data_cols + poll_list].reset_index(drop=True)
data_piv = pd.pivot_table(data_poll, index='sample_date', columns = 'id', values=poll_list).dropna()

p='N-NO3'
p='P-PO4'
for p in poll_list:
    print(p)
    p_piv = data_piv[p]

    diff = p_piv[3]- p_piv[1]
    diff.mean()

    diff_pct = (p_piv[1] - p_piv[3]) / p_piv[1] * 100
    diff_pct.mean()
    print(f'For chemical {p} average change from point 1 to 3: {np.round(diff.mean(),2)}')
    # diff.median()
    # diff.sem()

def analysisByUpperStream(df, savefolder, poll_list):

    data_cols = ['sample_date', 'id', 'stream_section']
    data_poll = df.loc[df.id<=3, data_cols+poll_list].reset_index(drop=True)


    data_melt = pd.melt(data_poll, id_vars=data_cols, value_vars=poll_list,var_name='element', value_name='conc').dropna().sort_values(by='conc',ascending=False)
    data_melt['perc'] = 0
    data_melt['n'] = 0

    # p='N-NH4'
    for p in poll_list:
        for id in data_melt.id.unique():
            data_melt.loc[(data_melt.element==p) & (data_melt.id==id), 'perc'] = data_melt.loc[(data_melt.element==p) & (data_melt.id==id), 'conc'].rank(pct=True) * 100
            data_melt.loc[(data_melt.element==p) & (data_melt.id==id), 'perc'] = 100 - data_melt.loc[(data_melt.element==p) & (data_melt.id==id), 'perc']
    #

    data_melt['perc'].max()


    # cmap = sns.color_palette(['#66c2a5','#fc8d62','#8da0cb','#e78ac3'])
    # cmap = sns.color_palette('muted', 6)
    # del cmap[2:4]
    cmap = sns.color_palette(["#7eb0d5", "#bd7ebe", "#ffb55a"])

    p = sns.relplot(data=data_melt, kind='line', x='perc', y='conc', col='element', hue='id', col_wrap=2, height=4, aspect=.8,
                    linewidth = 1.5, alpha=.75, facet_kws={'sharey': False, 'sharex': True}, palette=cmap)

    #set title, labels, and tics
    p.set_titles("{col_name}", weight='bold')
    p.set_axis_labels('% of samples above concentration', 'Concentration (mg/L)', weight='bold')
    p._legend.remove()

    title='Cl'
    ## add standards to graphs
    i=0
    for ax in p.axes.flat:
        title = ax.get_title()
        ax.set_xlim(-1,100)
        ax.set_xticks(np.arange(0,101,10))
        ax.set_xticks(np.arange(0,101,5), minor=True)
        ax.tick_params(labelleft=True)

        if title == 'EC':
            ax.set_ylabel('Electrical conductivity (mS/cm)', weight='bold')
        else:
            ax.set_ylabel('Concentration (mg/L)', weight='bold')

        try:
            std = standards[title]
            print(f'{title} has  standard: {std}')
            ax.axhline(std[0], ls='--', color='green')
            ax.axhline(std[1], ls='--', color='red')
        except:
            print(f'{title} has no standard')

    handles, labels = p.axes[0].get_legend_handles_labels()
    custom_lines = [Line2D([0], [0], color='green', ls='--'),
                    Line2D([0], [0], color='red', ls='--')]
    for line in custom_lines:
        handles.append(line)
    custom_labels = ['High quality\n[upper limit]', 'Low quality\n[lower limit]']
    for label in custom_labels:
        labels.append(label)

    # .legend(handles=handles, labels=labels, title='')
    p.add_legend(dict(zip(labels, handles)), title='')
    plt.subplots_adjust( wspace=0.25)

    p.savefig(f'{savefolder}concentration_by_id_points1-3.jpeg', bbox_inches='tight', dpi=300)

def timeSeries(df, savefolder, poll_list):

    id_list = [1,3,6,7,8,11]
    data_cols = ['sample_date', 'id', 'stream_section']
    data_poll = df.loc[df.id.isin(id_list), data_cols+poll_list].reset_index(drop=True)
    data_melt = pd.melt(data_poll, id_vars=data_cols, value_vars=poll_list,var_name='element', value_name='conc').dropna().sort_values(by='conc',ascending=False)
    data_melt['perc'] = 0

    cmap = sns.color_palette('muted', len(id_list))
    g = sns.FacetGrid(data=data_melt, col='element', col_wrap=2, height=4, aspect=.9, hue='id', sharey=False, sharex=True)
    g.map(sns.lineplot, 'sample_date', 'conc',  linewidth = 1.5, alpha=.75, palette=cmap, markers=True)
    # g.map(sns.pointplot, 'sample_date', 'conc',  ms='.', alpha=.75, palette=cmap)

    # p = sns.relplot(data=data_melt, kind='line', x='sample_date', y='conc', col='element', hue='id', col_wrap=2, height=4, aspect=.85,
    #                 linewidth = 1.5, alpha=.75, facet_kws={'sharey': False, 'sharex': True}, palette=cmap)

    #set title, labels, and tics
    g.set_titles("{col_name}", weight='bold')
    g.set_axis_labels('Sample date', 'Concentration (mg/L)', weight='bold')
    # try:
    #     g._legend.remove()
    # except:
    #     print("no legend to remove")

    title='Cl'
    ## add standards to graphs

    for ax in g.axes.flat:
        title = ax.get_title()
        myFmt = mdates.DateFormatter('%m/%y')
        ax.xaxis.set_major_formatter(myFmt)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        ax.tick_params(labelleft=True)

        if title == 'EC':
            ax.set_ylabel('Electrical conductivity (mS/cm)', weight='bold')
        else:
            ax.set_ylabel('Concentration (mg/L)', weight='bold')

        try:
            std = standards[title]
            print(f'{title} has  standard: {std}')
            ax.axhline(std[0], ls='--', color='green')
            ax.axhline(std[1], ls='--', color='red')
        except:
            print(f'{title} has no standard')
    handles, labels = g.axes[0].get_legend_handles_labels()
    labels = [f'Point {x}' for x in labels]
    custom_lines = [Line2D([0], [0], color='green', ls='--', label='High quality\n[upper limit]'),
                    Line2D([0], [0], color='red', ls='--', label='Low quality\n[lower limit]')]
    for line in custom_lines:
        handles.append(line)
    custom_labels = ['High quality\n[upper limit]', 'Low quality\n[lower limit]']
    for label in custom_labels:
        labels.append(label)

    plt.legend(handles=handles, labels=labels, frameon=False, bbox_to_anchor=(1,1.25), loc='lower left')
    plt.subplots_adjust( wspace=0.25)

    g.savefig(f'{savefolder}timeseries_points_{"_".join([str(x) for x in id_list])}.jpeg', bbox_inches='tight', dpi=300)

analysisByStreamSection(df, savefolder, poll_list)

