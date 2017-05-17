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





data = []
for i in range(1,21):
    best = 0
    temp = []
    for filename in os.listdir('data/ris/task_{}/'.format(i)):

        d = pickle.load( open("data/ris/task_{}/{}".format(i,filename), "rb" ) )
        v = float(filename[:-4].split("}")[1])
        if v > best:
            best = v
            temp = d
    data.append(temp)

plt.rc('text', usetex=True)
plt.rc('font', family='Times-Roman')
sns.set_style(style='white')
color = sns.color_palette("Set2", 10)
fig = plt.figure(figsize=(10,10))
i=1
for d in data:
    loss_train = [v for k,v in d[0].items()]
    loss_val = [v for k,v in d[2].items()]
    loss_test = [v for k,v in d[4].items()]

    acc_train = [v for k,v in d[1].items()]
    acc_val = [v for k,v in d[3].items()]
    acc_test = [v for k,v in d[5].items()]

    # print('loss_train:'+ str(min(loss_train)))
    # print('loss_val:'+ str(min(loss_val)))
    # print('acc_train:'+ str(max(acc_train)))
    # print('acc_val:'+ str(max(acc_val)))

    ax = fig.add_subplot(5,4, i)
    plt.title("Task "+str(i))
    plt.plot(acc_train, label=str(i))
    plt.plot(acc_val)
    if( i in [1,5,9,13,17]):
        ax.set_ylabel("Loss")
    if( i in [17,18,19,20]):
        ax.set_xlabel("Epoch")
    if(max(acc_test)>=0.94):
        ax.patch.set_facecolor("green")
        ax.patch.set_alpha(0.5)
    else:
        ax.patch.set_facecolor("red")
        ax.patch.set_alpha(0.5)
    i+=1

plt.tight_layout()
plt.show()
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
