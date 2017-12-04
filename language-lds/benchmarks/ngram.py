# n-gram model
import numpy as np

class NgramModel:
	def __init__(self, arity=3, dimension=None, laplace=0.1):
		if dimension == None:
			self.dimension = 100000 # just guess something, lol
		else:
			self.dimension = dimension
		self.arity = arity
		self.state = [-1] * (arity - 1)
		self.next = dict() # (n-1)-gram -> list of next words
		self.freqs = np.zeros(self.dimension + 1)
		self.laplace = laplace
		self.cache = dict()

	def fit(self, seq):
		n_distinct = 0

		# build data structure
		for i in range(len(seq) - self.arity + 1):
			prefix = tuple(seq[i:i + self.arity - 1])
			nextword = seq[i + self.arity - 1]
			if prefix in self.next:
				self.next[prefix].append(nextword)
			else:
				self.next[prefix] = [nextword]
				n_distinct += 1

		print '%d distinct %d-grams' % (n_distinct, self.arity)

		# also compute unigram freqs
		for w in seq:
			self.freqs[w] += 1
		self.freqs /= len(seq)

	def predict(self):
		prefix = tuple(self.state)

		# can't find? guess by unigrams
		if not prefix in self.next:
			return self.freqs

		# have I done this computation before?
		if prefix in self.cache:
			return self.cache[prefix]

		# otherwise, return Laplace-smoothed empirical n-gram dist
		ans = self.freqs * self.laplace * len(self.next[prefix])
		for w in self.next[prefix]:
			ans[w] += 1
		ans /= np.sum(ans)

		# cache if this query seems like it'll happen again
		if len(self.next[prefix]) > 10:
			self.cache[prefix] = ans

		return ans

	def step(self, w):
		self.state = self.state[1:] + [w]

if __name__ == '__main__':
	from ModelTester import ModelTester
	ModelTester(NgramModel(2, 13350, 0.8)).train().test()
