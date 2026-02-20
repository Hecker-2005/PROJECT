import numpy as np 

class FeatureSqueezer:
        """
        Deterministic feature squeezing for tabular NIDS data.
        Applied at inference time BEFORE model forward pass.
        """

        def __init__(
            self,
            feature_names,
            decimal_features = None,
            integer_features = None,
            clip_bounds = None
        ):
            """
            Parameters
            ----------
            feature_names : list[str]
                Ordered list of features (must match model input order)

            decimal_features : dict[str, int]
                Feature -> number of decimal places to keep

            integer_features : list[str]
                Features that should be rounded to nearest integer

            clip_bounds : dict[str, tuple]
                Feature -> (min, max) clipping bounds
            """
            self.feature_names = feature_names
            self.name_to_idx = {f: i for i, f in enumerate(feature_names)}

            self.decimal_features = decimal_features or {}
            self.integer_features = integer_features or []
            self.clip_bounds = clip_bounds or {}

        def squeeze(self, x):
            """
            Apply feature squeezing.

            Parameters
            ----------
            x : np.ndarray
                Shape (n_features,) or (n_samples, n_features)

            Returns
            -------
            np.ndarray
                Squeezed version of x
            """
            x = np.asarray(x, dtype = np.float64).copy()

            # Handle single sample
            single = False
            if x.ndim == 1:
                x = x.reshape(1, -1)
                single = True
            
            for feature, decimals in self.decimal_features.items():
                if feature in self.name_to_idx:
                    idx = self.name_to_idx[feature]
                    x[:, idx] = np.round(x[:, idx], decimals)

            for feature in self.integer_features:
                if feature in self.name_to_idx:
                    idx = self.name_to_idx[feature]
                    x[:, idx] = np.round(x[:, idx])

            for features, (low, high) in self.clip_bounds.items():
                if feature in self.name_to_idx:
                    idx = self.name_to_idx[feature]
                    x[:, idx] = np.clip(x[:, idx], low, high)
            
            return x[0] if single else x