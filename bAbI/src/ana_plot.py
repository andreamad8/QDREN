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









# d1 = pickle.load( open('checkpoints/grid_simple_CBT_NE_SIMPLE.pik', "rb" ) )
# d2 = pickle.load( open('checkpoints/grid_simple_CBT_NE_NORMAL.pik', "rb" ) )
#
#
# d1 =sorted([v for k,v in d1.items()])
# d2 =sorted([v for k,v in d2.items()])
#
# for e1,e2 in zip(d1,d2):
#     print(e1,e2)


plt.rc('text', usetex=True)
plt.rc('font', family='Times-Roman')
sns.set_style(style='white')
color = sns.color_palette("Set2", 10)
fig = plt.figure(figsize=(13,10))
i=1
for j in range(1):
    data = pickle.load( open('data/ris/task_2/{\'nb\': 20, \'tr\': [1, 1, 0, 0], \'L2\': 0.0, \'lr\': 0.001, \'dr\': 0.5, \'bz\': 32}', "rb" ) )
    print(data)
#     loss_train = [v for k,v in data[0].items()]
#     loss_val = [v for k,v in data[2].items()]
#     acc_train = [v for k,v in data[1].items()]
#     acc_val = [v for k,v in data[3].items()]
#     print(len(data))
#     print(len(loss_train))
#     print('loss_train:'+ str(max(loss_train)))
#     print('loss_val:'+ str(max(loss_val)))
#     print('acc_train:'+ str(max(acc_train)))
#     print('acc_val:'+ str(max(acc_val)))
#
#     ax1 = fig.add_subplot(2, 1, 1)
#     # ax1.set_xlim([0,20])
#     plt.plot(loss_train, label=str(i))
#     plt.plot(loss_val)
#
#     # ax2 = fig.add_subplot(2, 2, 2)
#     # ax2.set_xlim([0,20])
#     # ax2.set_ylim([0,4])
#
#     ax3 = fig.add_subplot(2, 1, 2)
#     # ax3.set_xlim([0,20])
#     plt.plot(acc_train)
#     plt.plot(acc_val)
#
#     # ax4 = fig.add_subplot(2, 2, 4)
#     # ax4.set_xlim([0,20])
#     ax1.legend(loc='lower center', bbox_to_anchor=(0.50, 0.43), bbox_transform=plt.gcf().transFigure)
#     i+=1
# plt.show()
#
















# data=[]
# for filename in os.listdir('../data/ris/task_2/'):
#     data.append([filename, pickle.load( open('../data/ris/task_2/' + filename, "rb" ) )])
#
# fig = plt.figure()
# i = 1
# for d in data:
#     loss_train = [v for k,v in d[1][0].items()]
#     loss_val = [v for k,v in d[1][2].items()]
#     x = [j for j in range(len(loss_train))]
#     ax = fig.add_subplot(3, 4, i)
#     plt.plot(loss_train)
#     plt.plot(loss_val)
#
#     i+=1
#
# plt.show()
#
#
# fig = plt.figure()
# i = 1
# for d in data:
#     acc_train = [v for k,v in d[1][1].items()]
#     acc_val = [v for k,v in d[1][3].items()]
#     x = [j for j in range(len(loss_train))]
#     ax = fig.add_subplot(3, 4, i)
#     plt.plot(acc_train)
#     plt.plot(acc_val)
#
#     i+=1
#
# plt.show()
