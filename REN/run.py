from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time
import datetime
from src.utils import Dataset
from src.EN import EntityNetwork
import numpy as np
import tensorflow as tf
import tflearn
import random
import docopt
import pickle





def main():
    batch_size = 32
    epoch = 2
    task_num = 1
    embedding_size = 100 # 100
    num_blocks = 2 # vocab_size
    lr = 0.001
    opt= "Adam"
    trainable = [0,0,0,0] # [E,T_E,K, T_K]
    sample_size = '' #'-10k'
    data= Dataset('data/tasks_1-20_v1-2/en-valid{}/'.format(sample_size),task_num,embedding_size)

    par={
        "vocab_size":data._data["vocab_size"],
        "sent_len":data._data['sent_len'],
        "sent_numb":data._data['sent_numb'],
        "embedding_size": embedding_size,
        "embeddings_mat": data._data['embeddings_mat'],
        "num_blocks":  data._data["vocab_size"],
        "learning_rate": lr,
        "clip_gradients":40.0,
        "opt":"Adam",
        "trainable": trainable,
        "max_norm": 1,
        "no_out": False,
        "decay_steps": epoch* data._data['len_training']/25,
        "decay_rate": 0.5
        }

    with tf.Session() as sess:
        entity_net = EntityNetwork(**par)
        print('Init stated')
        sess.run(tf.global_variables_initializer())
        train_loss, train_acc, val_loss, val_acc = {}, {}, {}, {}

        print('Training stated')
        best_val=0.0
        for e in range(epoch):
            loss, acc, counter = 0.0, 0.0, 0
            for i, elem in enumerate(data.get_batch_train(batch_size)):
                dic = data.get_dic_train(entity_net.S,entity_net.Q,entity_net.A,elem[0],elem[1])
                curr_loss, curr_acc, _ = sess.run([entity_net.loss_val, entity_net.accuracy, entity_net.train_op],
                                                  feed_dict=dic)
                loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
                if counter % 10 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss: %.3f\tTrain Accuracy: %.3f" % (e, counter, loss / float(counter), acc / float(counter)))
            # Add train loss, train acc to data
            train_loss[e], train_acc[e] = loss / float(counter), acc / float(counter)

            # Increment Epoch
            sess.run(entity_net.epoch_increment)


            if e % 10 == 0:
                dic = data.get_dic_val(entity_net.S,entity_net.Q,entity_net.A)
                val_loss_val, val_acc_val = sess.run([entity_net.loss_val, entity_net.accuracy], feed_dict=dic)
                print ("Epoch %d Validation Loss: %.3f\tValidation Accuracy: %.3f" % (e, val_loss_val, val_acc_val))

                # Add val loss, val acc to data
                val_loss[e], val_acc[e] = val_loss_val, val_acc_val

                # Update best_val
                if val_acc[e] > best_val:
                    best_val = val_acc[e]

                # # Save Model
                # saver.save(sess, ckpt_dir + "model.ckpt", global_step=entity_net.epoch_step)
                # with open(ckpt_dir + "training_logs.pik", 'w') as f:
                #     pickle.dump((train_loss, train_acc, val_loss, val_acc), f)

                # Early Stopping Condition
                if best_val > 0.95:
                    break
        # Test Loop
        dic = data.get_dic_test(entity_net.S,entity_net.Q,entity_net.A)
        test_loss, test_acc = sess.run([entity_net.loss_val, entity_net.accuracy], feed_dict=dic)

        # Print and Write Test Loss/Accuracy
        print ("Test Loss: %.3f\tTest Accuracy: %.3f" % (test_loss, test_acc))



if __name__ == '__main__':
    main()
