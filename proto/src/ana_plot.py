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
fig = plt.figure(figsize=(10,10))
i=1
for j in range(1):
    data = pickle.load( open('checkpoints/training_logs.pik'.format(j), "rb" ) )

    loss_train = [v for k,v in data[0].items()]
    loss_val = [v for k,v in data[2].items()]
    loss_test = [v for k,v in data[4].items()]

    acc_train = [v for k,v in data[1].items()]
    acc_val = [v for k,v in data[3].items()]
    acc_test = [v for k,v in data[5].items()]

    print('loss_train:'+ str(min(loss_train)))
    print('loss_val:'+ str(min(loss_val)))
    print('loss_test:'+ str(min(loss_test)))
    print('acc_train:'+ str(max(acc_train)))
    print('acc_val:'+ str(max(acc_val)))
    print('acc_test:'+ str(max(acc_test)))

    ax1 = fig.add_subplot(2, 1, 1)
    plt.title("Learning Curves")
    # ax1.set_xlim([0,20])
    plt.plot(loss_train,linewidth=1.95, alpha=0.7, color='gray',label='Training')
    plt.plot(loss_val,linewidth=1.95,linestyle='--',alpha=0.7, color='red',label='Validation')
    # plt.plot(loss_test)
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")


    ax2 = fig.add_subplot(2, 1, 2)
    # plt.title("Accuracy")

    # ax3.set_xlim([0,20])
    plt.plot(acc_train,linewidth=1.95, alpha=0.7, color='gray',label='Training')
    plt.plot(acc_val,linewidth=1.95,linestyle='--',alpha=0.7, color='red',label='Validation')
    # plt.plot(acc_test)
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")


    # ax4 = fig.add_subplot(2, 2, 4)
    # ax4.set_xlim([0,20])
    ax1.legend(loc='best')
    i+=1
# plt.savefig('checkpoints/data/FINAL_RIS/CNN_WIND/wind.pdf', format='pdf', dpi=300)
plt.show()

















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
