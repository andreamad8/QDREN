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


for i in range(1,21):
    temp = []
    for filename in os.listdir('data/ris/task_{}/'.format(i)):
        d = pickle.load( open("data/ris/task_{}/{}".format(i,filename), "rb" ) )
        temp.append(max([v for k,v in d[5].items()]))
    print('{}\t{}'.format(min(temp),max(temp)))


# data = []
# par  = ['']
# for i in range(1,21):
#     best = 0
#     for filename in os.listdir('data/ris/task_{}/'.format(i)):
#         d = pickle.load( open("data/ris/task_{}/{}".format(i,filename), "rb" ) )
#         a = max([v for k,v in d[5].items()])
#         if a > best:
#             best = a
#             temp = d
#             temp_par = filename.split('}')[0]+'}'
#     data.append(temp)
#     par.append(ast.literal_eval(temp_par))
# plt.rc('text', usetex=True)
# plt.rc('font', family='Times-Roman')
# sns.set_style(style='white')
# color = sns.color_palette("Set2", 10)
# fig = plt.figure(figsize=(10,10))
# i=1
# print('Task\tloss_train\tloss_val\tloss_test\tacc_train\tacc_val\tacc_test\tNB\tGL\tL2\tlr\tdr')
# for d in data:
#     loss_train = [v for k,v in d[0].items()]
#     loss_val = [v for k,v in d[2].items()]
#     loss_test = [v for k,v in d[4].items()]
#
#     acc_train = [v for k,v in d[1].items()]
#     acc_val = [v for k,v in d[3].items()]
#     acc_test = [v for k,v in d[5].items()]
#
#     idx = np.where(acc_val == max(acc_val))[0][-1]
#     print("%d\t%2f\t%2f\t%2f\t%2f\t%2f\t%2f\t%d\t%s\t%4f\t%4f\t%4f" % (i,loss_train[idx],loss_val[idx],
#                                                   loss_test[idx],acc_train[idx],
#                                                   acc_val[idx],acc_test[idx],
#                                                   int(par[i]['nb']),bool(par[i]['tr'][0]),
#                                                   float(par[i]['L2']),float(par[i]['lr']),float(par[i]['dr'])))
#
#     ax = fig.add_subplot(5,4, i)
#     plt.title("Task "+str(i))
#     plt.plot(acc_train, label=str(i))
#     plt.plot(acc_val)
#     if( i in [1,5,9,13,17]):
#         ax.set_ylabel("Accuracy")
#     if( i in [17,18,19,20]):
#         ax.set_xlabel("Epoch")
#     if(acc_test[idx] >= 0.95):
#         ax.patch.set_facecolor("green")
#         ax.patch.set_alpha(0.5)
#     else:
#         ax.patch.set_facecolor("red")
#         ax.patch.set_alpha(0.5)
#     i+=1
#
# plt.tight_layout()
# # plt.savefig('data/acc.pdf', format='pdf', dpi=300)
#
# plt.show()
