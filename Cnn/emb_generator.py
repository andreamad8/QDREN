import os
from nltk.tokenize import sent_tokenize
from nltk.tokenize import TweetTokenizer
import nltk
from gensim.models import Word2Vec
import codecs

class MySentences(object):
	def __init__(self, dirname_train,dirname_val,dirname_test):
		self.dirname_train = dirname_train
		self.dirname_val = dirname_val
		self.dirname_test = dirname_test

	def __iter__(self):
		for d in [self.dirname_train,self.dirname_val,self.dirname_test]:
			for fname in os.listdir(d):
				data=[]
				filename=os.path.join(d, fname)
				for line in codecs.open(filename,'r', 'utf-8'):
					data.append(line[:-1])
				story_ent    = sent_tokenize(data[2])
				question_ent = sent_tokenize(data[4])
				story_ent = [s.lower().split() for s in story_ent]
				question_ent = [s.lower().split() for s in question_ent]
				ans = data[6].replace(" ", "_")
				ent = {data[i].split(":")[0]: data[i].split(":")[1].lower()  for i in range(8,len(data))}

				for q_sent in question_ent:
					for i in range(len(q_sent)):
						if q_sent[i] == "@placeholder":
							q_sent[i] = ent[ans].replace(" ", "_")

				for s_sent in story_ent:
					for i in range(len(s_sent)):
						if(s_sent[i] in ent):
							s_sent[i]=ent[s_sent[i]].replace(" ", "_")
				sent= story_ent+question_ent
				for s in sent:
					yield s

sentences = MySentences('data/cnn/questions/training','data/cnn/questions/validation','data/cnn/questions/test') # a memory-friendly iterator

### sg= CBOW
model = Word2Vec(sentences, size=100, window=5,min_count=1, workers=8,sg=0)
print (model)
model.wv.save_word2vec_format('data/emb_100_CBOW_wind_5.txt', binary=False)
