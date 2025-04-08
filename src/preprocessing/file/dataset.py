import numpy as np

class DataSet:
    numeric_data_index = 4

    def __init__(self, data: np.ndarray, headers: list[str]):
        self._data = data
        self.headers = headers

    def get_numerics(self):
        """Return only the columns that are numerical numbers"""
        return self._data[:, self.numeric_data_index:]

    def get_column(self, name: str):
        return self._data[:, self.headers.index(name)]

    def get_all(self):
        return self._data