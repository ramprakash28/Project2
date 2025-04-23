import numpy as np

class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1:
            return {'class': y[0]}

        if depth >= self.max_depth:
            return {'class': np.bincount(y.astype(int)).argmax()}

        best_gini = float('inf')
        best_split = None
        best_left_mask = None
        best_right_mask = None

        n_samples, n_features = X.shape
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                left_y = y[left_mask]
                right_y = y[right_mask]

                gini = self._gini_index(left_y, right_y)
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_idx, threshold)
                    best_left_mask = left_mask
                    best_right_mask = right_mask

        left_tree = self._build_tree(X[best_left_mask], y[best_left_mask], depth + 1)
        right_tree = self._build_tree(X[best_right_mask], y[best_right_mask], depth + 1)

        return {
            'feature_idx': best_split[0],
            'threshold': best_split[1],
            'left': left_tree,
            'right': right_tree
        }

    def _gini_index(self, left_y, right_y):
        def gini(y):
            if len(y) == 0:
                return 0
            classes = np.bincount(y.astype(int))
            probs = classes / len(y)
            return 1 - np.sum(probs ** 2)

        left_gini = gini(left_y)
        right_gini = gini(right_y)
        total = len(left_y) + len(right_y)
        return (len(left_y) / total) * left_gini + (len(right_y) / total) * right_gini

    def predict_one(self, x, node=None):
        if node is None:
            node = self.tree
        if 'class' in node:
            return node['class']
        if x[node['feature_idx']] <= node['threshold']:
            return self.predict_one(x, node['left'])
        else:
            return self.predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        y = y.astype(float)
        F = np.zeros_like(y)
        for _ in range(self.n_estimators):
            residuals = y - self._sigmoid(F)
            model = DecisionTree(max_depth=self.max_depth)
            model.fit(X, residuals)
            predictions = model.predict(X)
            F += self.learning_rate * predictions
            self.models.append(model)

    def predict(self, X):
        F = np.sum([self.learning_rate * model.predict(X) for model in self.models], axis=0)
        return (self._sigmoid(F) >= 0.5).astype(int)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
