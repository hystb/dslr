import src.maths as mts
import numpy as np

class Z_score():
    def normalize_matrix(self, data: np.ndarray):
        normalized_data = data.copy()

        for i in range(data.shape[1]):
            col = data[:, i]
            mean = mts.mean(col)
            std = mts.std(col)

            normalized_data[:, i] = [self.normalize_column(x, mean, std) for x in col]
        return normalized_data

    def normalize_column(self, x, mean, std):
        return (x.astype(float) - mean) / std
