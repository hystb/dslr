import numpy as np

from src.preprocessing.scalers import Z_score

class DataSet:
    numeric_data_index = 4 # with birthday and besthand !

    def __init__(self, data: np.ndarray, headers: dict):
        self._data = data
        self._headers = headers

    def get_numerics(self, exclude: list[str] = []) -> np.ndarray:
        """Return only the columns that are numerical numbers"""
        array = self._data[:, self.numeric_data_index:]
        idx_del = [self._headers.get(ex) - self.numeric_data_index for ex in exclude]
        return np.delete(array, idx_del, axis=1)
    
    def get_column(self, name: str) -> np.ndarray:
        return self._data[:, self._headers.get(name)]

    def get_all(self) -> np.ndarray:
        return self._data

    def get_headers(self) -> dict:
        return self._headers
    
    def standardize(self):
        """
        Return a standardize instance of the current dataset.
        Replacer must not NoneReplacer!
        """
        score = Z_score()

        z_data = self._data.copy()
        z_data[:, self.numeric_data_index:] = score.normalize_matrix(self.get_numerics())

        return DataSet(z_data, self._headers)
