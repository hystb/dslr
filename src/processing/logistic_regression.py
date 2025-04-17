import numpy as np

class LogisticRegressionModel:
    """LogisticRegression using BinaryCrossEntropy (loss) and batch descent gradient (optimizer)"""

    def __init__(self, learning_rate = 0.01):
        self.learning_rate = learning_rate

    @staticmethod
    def loss(A, Y):
        """
        A = result of the activation function
        Y = real predictions labels
        """
        epsilon = 1e-15
        A = np.clip(A, epsilon, 1 - epsilon)
        # as mentionned in the subject
        return (-1 / A.shape[0]) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    @staticmethod
    def activation(x):
        """Sigmoide activation function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def linear(X, weights, bias) -> float:
        return np.dot(X, weights) + bias

    @classmethod
    def y_prediction(cls, X, weights, bias) -> np.ndarray:
        return cls.activation(cls.linear(X, weights, bias))

    @classmethod
    def early_stop(cls, loss_history: list):
        delta = 1e-4

        if (len(loss_history) > 2):
            if (loss_history[-2] - loss_history[-1] < delta):
                return True
            else:
                return False
        else:
            return False

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        n_samples, n_features = X.shape
        loss_history = []
        self.weights = np.zeros(n_features)
        self.bias = 0

        while not (self.early_stop(loss_history)):
            y_predicted = self.y_prediction(X, self.weights, self.bias)

            # as mentionned in subject
            dw = 1 / n_samples * np.dot(X.T, (y_predicted - y))
            db = 1 / n_samples * np.sum(y_predicted - y)

            # gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            loss_history.append(self.loss(self.y_prediction(X, self.weights, self.bias), y))

        return loss_history[-1] 

    def predict(self, X):
        return self.y_prediction(X, self.weights, self.bias)

    def load(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def export(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.weights, self.bias)

class OneVsAllLogisticRegression():
    class_indexes = {
        "Ravenclaw": 0,
        "Slytherin": 1,
        "Gryffindor": 2,
        "Hufflepuff": 3
    }

    def __init__(self):
        self.models = {}

    @classmethod
    def _class_str_to_int(cls, y) -> np.ndarray:
        for k, v in cls.class_indexes.items():
            y[y == k] = v
        return np.array(y, dtype=int)

    def fit(self, X, y, **kwargs):
        """Automaticly perform one-vs-all regression"""
        y = self._class_str_to_int(y.copy())

        for k, v in self.class_indexes.items():
            binary_y = y.copy()
            binary_y = (binary_y == v).astype(int)
            model = LogisticRegressionModel(**kwargs)

            loss = model.fit(X, binary_y)
            self.models[k] = model

            print(f"Loss for model of {k} house: {loss:.3f}")

    def predict(self, X) -> str:
        """Return the predicted class name"""
        predictions = {
            name: model.predict(X) for name, model in self.models.items()
        }
        return max(predictions, key=predictions.get)
