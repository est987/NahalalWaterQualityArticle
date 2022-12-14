import seaborn as sns
from matplotlib import rcParams,use

#setStyle
def setStyle():
    sns.set_style("whitegrid", {'font.family':'Calibri','xtick.bottom': True, 'ytick.left':True, 'xtick.color':'black', 'ytick.color':'black', 'legend.frameon': True, \
                                'axes.grid': False, 'axes.spines.right':False,'axes.spines.top':False})
    sns.despine()
    sns.axes_style()
    rcParams['axes.labelsize'] = 14
    rcParams['xtick.labelsize'] = 11
    rcParams['ytick.labelsize'] = 11
    rcParams['grid.alpha'] = 0.4
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['axes.edgecolor'] = 'black'
    rcParams['figure.titleweight'] = 'bold'
    rcParams['axes.labelweight'] = 'bold'
    rcParams['font.weight'] = 'normal'
    rcParams['font.size'] = 11
    rcParams['font.family'] = 'Calibri'

## change matplotlib graphrender
def changeMPLBackend():
    # use('tkAgg')
    use('qt5Agg')