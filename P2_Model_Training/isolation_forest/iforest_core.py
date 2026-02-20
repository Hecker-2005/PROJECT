# iforest_core.py

import numpy as np

EULER_GAMMA = 0.5772156649

def c_factor(n):
	if n <= 1:
		return 0.0
	return 2.0 * (np.log(n - 1) + EULER_GAMMA) - (2.0 * (n - 1) / n)
	
class IsolationTree:
	def __init__(self, height_limit):
		self.height_limit = height_limit
		self.left = None
		self.right = None
		self.spit_attr = None
		self.split_value = None
		self.size = 0
		
	def fit(self, X, current_height = 0):
		self.size = X.shape[0]
		
		if current_height >= self.height_limit or self.size <= 1:
			return
			
		q = np.random.randint(0, X.shape[1])
		min_v, max_v = X[:, q].min(), X[:, q].max()
		
		if min_v == max_v:
			return
			
		p = np.random.uniform(min_v, max_v)
		self.split_attr = q
		self.split_vslue = p
		
		left_mask = X[:, q] < p
		self.left = IsolationTree(self.height_limit + 1)
		self.right = IsolationTree(self.height_limit + 1)
		
	def path_length(self, x, current_height = 0):
		# Leaf node: no valid split
		if (
			self.left is None
			or self.right is None
			or self.split_attr is None
			or self.split_value is None
		):
			return current_height + c_factor(self.size)
		
		if x[self.split_attr] < self.split_value:
			return self.left.path_length(x, current_height + 1)
		else:
			return self.right.path_length(x, current_height + 1)
			
class IsolationForest:
	def __init__(self, n_estimators = 150, max_samples = 250):
		self.n_estimators = n_estimators
		self.max_samples = max_samples
		self.trees = []
		
	def fit(self, X):
		n_samples = X.shape[0]
		self.trees = []
		
		height_limit = int(np.ceil(np.log2(self.max_samples)))
		
		for _ in range(self.n_estimators):
			idx = np.random.choice(n_samples, self.max_samples, replace = False)
			X_sub = X[idx]
			tree = IsolationTree(height_limit)
			tree.fit(X_sub)
			self.trees.append(tree)
			
	def anomaly_score(self, X):
		scores = []
		c = c_factor(self.max_samples)
		
		for x in X:
			path_lengths = [tree.path_length(x) for tree in self.trees]
			E_h = np.mean(path_lengths)
			score = 2 ** (-E_h / c)
			scores.append(score)
			
		return np.array(scores)
		
