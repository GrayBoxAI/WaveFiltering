# currently tests model on Brown corpus

import numpy as np

class ModelTester:
	def __init__(self, model):
		self.model = model

	def train(self):
		train_vec = np.load('../corpus/penn/train.npy')
		self.model.fit(train_vec)
		return self

	def test(self, ignore=0):
		test_vec = np.load('../corpus/penn/test.npy')
		perplexity = 0
		steps = 0
		for i,w in enumerate(test_vec):
			pred = self.model.predict() # guess the next word

			if i >= ignore:
				perplexity += np.log( 1/pred[w] ) # truth was how surprising?
				steps += 1
				if steps % 500 == 0:
					print 'perplexity (t=%d) = %.4f' % (i, np.exp(perplexity / steps))

			self.model.step(w) # next word please
		perplexity /= steps

		print 'perplexity:', np.exp(perplexity)
		return self