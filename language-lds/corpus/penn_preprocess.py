# taken from https://gist.github.com/zermelozf/5549143
import nltk
import numpy as np
import pickle
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

penn = nltk.corpus.treebank
corpus = [word.lower() for word in penn.words()]

# use 95% of corpus for training, 5% for testing
spl = 95*len(corpus)/100
train = corpus[:spl]
test = corpus[spl:]

# get vocab from wordvecs
wordvec_vocab = pickle.load(open('../wordvecs/glove.6B.300d/vocab.pickle'))

# Remove rare words from the corpus
fdist = nltk.FreqDist(w for w in train)
vocabulary = set(map(lambda x: x[0], filter(lambda x: x[1] >= 3
	and x[0] in wordvec_vocab, fdist.iteritems())))

# map to words
word2idx, idx2word = dict(), dict()
for i,w in enumerate(vocabulary):
	word2idx[w] = i
	idx2word[i] = w

train = map(lambda x: word2idx[x] if x in vocabulary else -1, train)
test = map(lambda x: word2idx[x] if x in vocabulary else -1, test)

print 'vocab size:', len(vocabulary)
print 'train size:', len(train)
print 'test size:', len(test)

# print train
# print test
# print vocabulary

np.save('penn/train.npy', np.array(train))
np.save('penn/test.npy', np.array(test))

with open('penn/vocab.pickle', 'wb') as handle:
	pickle.dump( (word2idx, idx2word), handle, protocol=pickle.HIGHEST_PROTOCOL)
