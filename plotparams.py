"""
One plot to rule them all
One plot to find them
One plot to bring them all
And in the darkness bind them
"""


import matplotlib.pyplot as plt
from cycler import cycler

plt.style.use('ggplot')
params = {'legend.fontsize': 10,
         'axes.labelsize': 15,
         'axes.labelpad': 4.0,
         'axes.titlesize': 20,
         'axes.labelcolor':'black',
         'lines.linewidth': 2.5,
         'lines.markersize':8,
         'xtick.labelsize': 13,
         'ytick.labelsize':13,
         'xtick.major.width': 6,
         'ytick.major.width': 6,
         'axes.prop_cycle'    : cycler('color',
                            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                              '#bcbd22', '#17becf'])
 }

plt.rcParams.update(params)
