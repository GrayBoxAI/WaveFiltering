# load word vectors

import numpy as np
import pickle
from annoy import AnnoyIndex

class GloveLoader:
	@staticmethod
	def build(vocab_path='../corpus/penn', vec_path='glove.6B.300d'):
		word2idx, idx2word = pickle.load( open(vocab_path + '/vocab.pickle') )
		vecs = [None] * len(word2idx)
		n_loaded = 0
		dim = 0

		# scan through all word vectors, isolating ones we need
		for i,line in enumerate(open(vec_path + '/vecs.txt')):
			items = line.split()
			word = items[0]
			vec = [float(x) for x in items[1:]]

			# does it match something in our corpus?
			if word.lower() in word2idx and vecs[word2idx[word.lower()]] == None:
				vec = np.array(vec)
				vec /= np.linalg.norm(vec)
				vecs[word2idx[word.lower()]] = vec
				if n_loaded == 0: # saw first vector. infer dimension
					dim = len(vec)
				n_loaded += 1

			# print progress: (words matched)/(words scanned)
			if (i+1) % 10000 == 0:
				print 'Loading words... (%d/%d)' % (n_loaded, i+1)

		# report unknowns
		n_unmatched = 0
		for i in range(len(vecs)):
			if vecs[i] == None:
				print idx2word[i]
				n_unmatched += 1
				vecs[i] = np.zeros(dim)
		print '%d words matched, %d unmatched' % (n_loaded, n_unmatched)

		return np.array(vecs)

if __name__ == '__main__':
	vecs = GloveLoader.build()
	print(vecs.shape)
	np.save('../corpus/penn/vecs.npy', vecs)

