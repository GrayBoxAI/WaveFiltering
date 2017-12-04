# extract vocab from pre-trained word vectors

import pickle

def extract_vocab(vec_path='glove.6B.300d'):
	words = []
	for i,line in enumerate(open(vec_path + '/vecs.txt')):
		items = line.split()
		word = items[0]
		words.append(word)
		if i % 10000 == 0:
			print i
	return words

if __name__ == '__main__':
	path = 'glove.6B.300d'
	words = extract_vocab(path)
	with open(path + '/vocab.pickle', 'wb') as handle:
		pickle.dump( set(words), handle, protocol=pickle.HIGHEST_PROTOCOL )
