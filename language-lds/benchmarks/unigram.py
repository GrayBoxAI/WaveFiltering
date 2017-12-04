# minimal toy example (to demonstrate API): 1-gram model (predict same distribution)
import numpy as np

class UnigramModel:
	def __init__(self, dimension=None):
		if dimension == None:
			self.dimension = 100000 # just guess something, lol
		else:
			self.dimension = dimension
		self.freqs = np.zeros(self.dimension + 1) # add -1 = "unknown"

	def fit(self, seq):
		'''Train the language model. In: sequence of word indices.'''
		for w in seq:
			self.freqs[w] += 1.0
		self.freqs /= len(seq)
		assert(abs(self.freqs.sum() - 1) <= 1e-10)

	def predict(self):
		'''Test the language model. Out: a probability vector, indexed in the
		same space as the input word indices.'''

		# predict the same distribution every round!
		return self.freqs

	def step(self, w):
		'''Advance by one state for testing. In: one word index.'''
		pass

if __name__ == '__main__':
	from ModelTester import ModelTester
	ModelTester(UnigramModel(13350)).train().test()
