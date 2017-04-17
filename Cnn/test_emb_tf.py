import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
import gensim
import os
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
import nltk
import codecs
import sys
from tqdm import tqdm
import cPickle as pickle

def get_data(dirname,file_lists,embeddings_index,max_sent=20,embedding_size=100,n_entities=700):
    S = []
    Q = []
    A = []
    for fname in tqdm(os.listdir(dirname)):
        data=[]
        filename=os.path.join(dirname, fname)
        for line in codecs.open(filename,'r', 'utf-8'):
	           data.append(line[:-1])

        story_ent    = [s.lower().split() for s in sent_tokenize(data[2])]
        question_ent = [s.lower().split() for s in sent_tokenize(data[4])]
        ans = data[6].replace(" ", "_")


        ent = {data[i].split(":")[0]: data[i].split(":")[1].lower()  for i in range(8,len(data))}
        question_ent= [question_ent[0]]
        for j in range(len(question_ent)):
            for i in range(len(question_ent[j])):
                if question_ent[j][i] == "@placeholder":
                    question_ent[j][i] = embeddings_index[ "<UNK>".decode('utf-8')]
                else:
                    question_ent[j][i] = embeddings_index[question_ent[j][i]]
            question_ent[j] = np.sum(np.asarray(question_ent[j]),0)


        k = 0
        ris_ent_sorted_by_postion = {v:[]for k,v in ent.items()}
        for j in range(len(story_ent)):
            for i in range(len(story_ent[j])):
                if(story_ent[j][i] in ent):
                    ris_ent_sorted_by_postion[ent[story_ent[j][i]]].append(k)
                    if(ent[story_ent[j][i]]!=''):
                        story_ent[j][i] = embeddings_index[ent[story_ent[j][i]].replace(" ", "_")]
                    else:
                        story_ent[j][i] = np.zeros(embedding_size)
                    k += 1
                else:
                    if(story_ent[j][i]!=''):
                        story_ent[j][i] = embeddings_index[story_ent[j][i]]
                    else:
                        story_ent[j][i] = np.zeros(embedding_size)

            story_ent[j] = np.sum(np.asarray(story_ent[j]),0)
        for i in range(len(story_ent),max_sent):
            story_ent.append(np.zeros(embedding_size))

        temp={}
        for k,v in ris_ent_sorted_by_postion.items():
            if(len(v)>0):
                temp[k]=min(v)
        temp= sorted(temp, key=temp.get)
        for i in range(len(temp)):
            ris_ent_sorted_by_postion[temp[i]] = i
        index_anw = ris_ent_sorted_by_postion[ent[ans]]
        ans_vec = np.zeros(n_entities)
        ans_vec[index_anw]=1

        if(np.array(question_ent).shape[0]!= 1):
            print(data[4])
            break
        # print(np.array(story_ent).shape)
        S.append(np.array(story_ent[:max_sent]))
        Q.append(np.array(question_ent))
        A.append(ans_vec)


    return np.array(S),np.array(Q),np.array(A)





def get_vocab_and_index(emb_file='data/emb_100_CBOW_wind_5.txt'):
    f = codecs.open(emb_file, "r", "utf-8")
    # load matrix
    embeddings_index = {}
    vocab = []
    embeddings_index["<UNK>".decode('utf-8')]= np.zeros(100)
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        vocab.append(word)
        coefs = np.asarray(values[1:]).astype(np.float)
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index


embeddings_index = get_vocab_and_index()

file_lists = os.listdir('data/cnn/questions/training')
S,Q,A = get_data('data/cnn/questions/training',
                    file_lists,
                    embeddings_index,
                    max_sent=20,
                    embedding_size=100,
                    n_entities=550)
pickle.dump( [S,Q,A] , open( "data/cnn/training_set.p", "wb" ) )

# get_vocab_and_index()






# embed_size = 100
# VOCAB =
# model =  gensim.models.Word2Vec.load('data/glove.6B.100d.txt')
# # prepare embedding matrix
# embedding_matrix = np.zeros((VOCAB, embed_size))
# for word, i in tokenizer.word_index.items():
#     if word in model:
#         embedding_vector = model[word]
#         if embedding_vector is not None:
#             # words not found in embedding index will be all-zeros.
#             embedding_matrix[i] = embedding_vector
#     #else:
#     #    print('Missing from QAEMB: {}'.format(word))
#
# print('Total number of null word embeddings:')
# print(np.sum(np.sum(embedding_matrix, axis=1) == 0))
