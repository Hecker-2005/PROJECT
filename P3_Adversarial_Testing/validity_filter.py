import numpy as np

class ValidityFilter:
	""" 
	Hard validity gate for tabular NIDS adversarial samples
	A sample is either VALID or DISCARDED
	"""
	
	def __init__(
		self,
		feature_names,
		immutable_features,
		integer_features,
		non_negative_features,
		dependency_rules
	):
		self.feature_names = feature_names
		self.name_to_idx = {f: i for i, f in enumerate(feature_names)}
		
		self.immutable_idx = [self.name_to_idx[f] for f in immutable_features]
		self.integer_idx = [self.name_to_idx[f] for f in integer_features]
		self.non_negative_idx = [self.name_to_idx[f] for f in non_negative_features]
		
		self.dependency_rules = dependency_rules

	def project(self, x_adv, x_orig):
		"""
		Project x_adv back into the feasible space.
		Used inside iterative attacks (FGSM / CAPGD).
		"""
		
		x = x_adv.copy()
		
		# Immutable features: restore original values
		for idx in self.immutable_idx:
			x[idx] = x_orig[idx]
			
		# Enforce non-negativity
		for idx in self.non_negative_idx:
			if x[idx] < 0:
				x[idx] = 0.0
				
		# Enforce integer-valued features
		for idx in self.integer_idx:
			x[idx] = int(round(x[idx]))
			
		return x
	
	def is_valid(self, x):
		"""
		Strict validity check.
		Returns True if sample is semantically valid.
		"""
		# Non-negativity
		for idx in self.non_negative_idx:
			if x[idx] < 0:
				return False

		# Integer integrity
		for idx in self.integer_idx:
			if not float(x[idx]).is_integer():
				return False

		# Dependency rules
		for rule_fn in self.dependency_rules:
			if not rule_fn(x, self.name_to_idx):
				return False

		return True
