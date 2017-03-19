from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time
from src.utils import Dataset
from mem_cell import DynamicMemoryCell
from functools import partial
from entNet import entNet
import numpy as np
import tensorflow as tf
import random



# config = tf.ConfigProto(log_device_placement=False)
# config.gpu_options.allow_growth = True
sess = tf.InteractiveSession()

data= Dataset('data/tasks_1-20_v1-2/en-valid/')

par={
    "vocab_size":data._data["vocab_size"],
    "sent_len":data._data['sent_len'],
    "sent_numb":data._data['sent_numb'],
    "embedding_size": 10,
    "num_blocks": 20,
    "learning_rate":1e-2,
    "clip_gradients":40.0,
    "opt":"Adam",
    "debug":True
    }


#### MODEL
accuracy,prediction,loss,train_op,S_input, Q_input, A_input = entNet(**par)



merged = tf.merge_all_summaries()
summary_writer_train = tf.train.SummaryWriter('logs/train', sess.graph)
summary_writer_val = tf.train.SummaryWriter('logs/val')


sess.run(tf.initialize_all_variables())
for step in range(14):
    for i,j in data.get_batch_train(9000):
        dic = data.get_dic_train(S_input,Q_input,A_input,i,j)
        summary,_, loss_train, acc_train = sess.run([merged,train_op, loss, accuracy], feed_dict=dic)
        summary_writer_train.add_summary(summary,step)
    if step % 10 == 0:
        dic = data.get_dic_val(S_input,Q_input,A_input)
        su,loss_val, acc_val = sess.run([merged,loss, accuracy], feed_dict=dic)
        summary_writer_val.add_summary(su,step)
        print ('Step {}: loss_train = {:.2f}, acc_train = {:.2f} loss_val = {:.2f}, acc_val = {:.2f}'.format(step, loss_train, acc_train,loss_val, acc_val))


dic = data.get_dic_test(S_input,Q_input,A_input)
loss_test, acc_test = sess.run([loss, accuracy], feed_dict=dic)
print ('loss_test = {:.2f}, acc_test = {:.2f}'.format(loss_test, acc_test))
summary_writer_train.close()
summary_writer_val.close()
