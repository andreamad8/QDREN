import json
import os
import numpy as np
import scipy as sp
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import ast



def import_data(filename):
    fo = open(filename, "r")
    data=['empty']
    for line in fo:
        data.append(line[:-1])
    fo.close()
    return data[len(data)-1]

data_1k=[]
for filename in os.listdir('../ris/'):
    if('10K.txt'in filename):
        if(filename[6:7]=='_'):
            data_1k.append([int(filename[5:6]),import_data('../ris/'+filename)])
        else:
            data_1k.append([int(filename[5:7]),import_data('../ris/'+filename)])


error=[]
for d in sorted(data_1k):
    error.append((1-float(d[1][len(d[1])-4:]))*100)
    print ('Task_{} {} {}'.format(d[0], d[1],(1-float(d[1][len(d[1])-4:]))*100 ))

print(sum(error)/20)
