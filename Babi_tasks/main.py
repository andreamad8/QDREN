"""
Usage:
    main.py [--batch_size=<kN>] [--embedding_size=<kN>] [--num_blocks=<kN>] [--epoch=<kN>] [--task_num=<kN>] [--learning_rate=<kN>] [--sample_size=<kN>]
    main.py -h | --help

Options:
    -h --help              Show this screen.
    --batch_size=<kN>      batch_size [default: 900].
    --embedding_size=<kN>  embedding_size [default: 50].
    --num_blocks=<kN>      num_blocks [default: 20].
    --epoch=<kN>           epoch  [default: 200].
    --task_num=<kN>        task_num   [default: 1].
    --learning_rate=<kN>   learning_rate  [default: 1e-2].
    --sample_size=<kN>     sample_size choose between 1K and 10K [default: 1].
"""
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
import docopt


def main(arguments):
    # config = tf.ConfigProto(log_device_placement=False)
    # config.gpu_options.allow_growth = True
    print(arguments)
    batch_size = arguments['--batch_size']
    embedding_size = arguments['--embedding_size']
    num_blocks = arguments['--num_blocks']
    epoch =  arguments['--epoch']
    task_num =  arguments['--task_num']
    learning_rate = arguments['--learning_rate']
    sample_size = '' if arguments['--sample_size']==1 else '-10k'
    data= Dataset('data/tasks_1-20_v1-2/en-valid{}/'.format(sample_size),task_num,embedding_size)
    par={
        "vocab_size":data._data["vocab_size"],
        "sent_len":data._data['sent_len'],
        "sent_numb":data._data['sent_numb'],
        "embedding_size": embedding_size,
        "embeddings_mat": data._data['embeddings_mat'],
        "num_blocks": data._data["vocab_size"],
        "learning_rate": learning_rate,
        "clip_gradients":40.0,
        "opt":"Adam",
        "debug":False
        }

    #### MODEL
    accuracy,prediction,loss,train_op,S_input, Q_input, A_input, output= entNet(**par)


    sess = tf.InteractiveSession()

    merged = tf.merge_all_summaries()
    summary_writer_train = tf.train.SummaryWriter('logs/train/task_{}{}'.format(str(task_num),sample_size), sess.graph)
    summary_writer_val = tf.train.SummaryWriter('logs/val/task_{}{}'.format(str(task_num),sample_size))


    sess.run(tf.initialize_all_variables())
    for step in range(epoch):
        te_loss=0
        for i, elem in enumerate(data.get_batch_train(batch_size)):
            dic = data.get_dic_train(S_input,Q_input,A_input,elem[0],elem[1])
            # print(dic)
            summary,_, loss_train, acc_train = sess.run([merged,train_op, loss, accuracy], feed_dict=dic)
            summary_writer_train.add_summary(summary,step*(data.num_batches)+i)
            te_loss+=loss_train
        loss_batch=te_loss/ data.len_train


        if step % 10 == 0:
            dic = data.get_dic_val(S_input,Q_input,A_input)
            su,loss_val, acc_val = sess.run([merged,loss, accuracy], feed_dict=dic)
            summary_writer_val.add_summary(su,step*data.num_batches)
            print ('Step {}: loss_train = {:.2f}, acc_train = {:.2f} loss_val = {:.2f}, acc_val = {:.2f}'.format(step, loss_batch, acc_train,loss_val/ data.len_val, acc_val))


    dic = data.get_dic_test(S_input,Q_input,A_input)
    loss_test, acc_test = sess.run([loss, accuracy], feed_dict=dic)
    print ('loss_test = {:.2f}, acc_test = {:.2f}'.format(loss_test/ data.len_test, acc_test))
    summary_writer_train.close()
    summary_writer_val.close()



if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    arguments= {k: int(v) if k!='--learning_rate' else float(v) for k, v in arguments.items()}
    main(arguments)
