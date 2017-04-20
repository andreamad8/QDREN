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
import cPickle as pickle


data=[]
for filename in os.listdir('../data/ris/task_2/'):
    data.append([filename, pickle.load( open('../data/ris/task_2/' + filename, "rb" ) )])

fig = plt.figure()
i = 1
for d in data:
    loss_train = [v for k,v in d[1][0].items()]
    loss_val = [v for k,v in d[1][2].items()]
    x = [j for j in range(len(loss_train))]
    ax = fig.add_subplot(3, 4, i)
    plt.plot(loss_train)
    plt.plot(loss_val)

    i+=1

plt.show()


fig = plt.figure()
i = 1
for d in data:
    acc_train = [v for k,v in d[1][1].items()]
    acc_val = [v for k,v in d[1][3].items()]
    x = [j for j in range(len(loss_train))]
    ax = fig.add_subplot(3, 4, i)
    plt.plot(acc_train)
    plt.plot(acc_val)

    i+=1

plt.show()
