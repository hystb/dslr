from src.preprocessing.file.replacer import AbstractReplacer, NoneReplacer
from src.preprocessing.file.dataset import DataSet
from datetime import datetime
import numpy as np
import csv

class Loader:
    birtday_c_index = 4
    besthand_c_index = 5

    @classmethod
    def load(cls, filepath: str, replacer = NoneReplacer) -> DataSet:
        """
        Create a dataset object from a csv.file

        `data treatment`:
        - empty values of the dataset will be handled by `replacer` object
        - birthday values are converted to ages (years)
        - best hand is converted to int left/right -> 0/1
        """
        with open(filepath, 'r') as file:
            data = list(csv.reader(file, delimiter=","))

            headers = {k: v for v, k in enumerate(data.pop(0))}
            data_array = cls.__to_matrix(data)

            data_treated = cls.data_treatment(data_array.copy(), replacer)

            return DataSet(data_treated, headers)

    @classmethod
    def data_treatment(cls, data: np.ndarray, replacer: AbstractReplacer) -> np.ndarray:
        data[:, cls.birtday_c_index] = cls.__convert_birthday(data[:, cls.birtday_c_index])
        data[:, cls.besthand_c_index] = cls.__convert_besthand(data[:, cls.besthand_c_index])

        data = cls.__convert_to_numbers(data)
        empty_dict = cls.__extract_empty_values(data)

        data = cls.replacement(empty_dict, data, replacer)

        return data

    @classmethod
    def replacement(cls, empty_values: dict, data, replacer: AbstractReplacer) -> np.ndarray:
        result = data.copy()

        for key, value in empty_values.items():
            original_column: np.ndarray = data.T[key]
            original_without_missing = np.delete(original_column, value)

            for index in value:
                # replacement method
                result.T[key][index] = replacer.replace(original_without_missing)

        return result

    @classmethod
    def __to_matrix(cls, data) -> np.ndarray:
        return np.array(data)

    @classmethod
    def __convert_to_numbers(cls, data) -> np.ndarray:
        def try_number(value):
            try:
                return float(value)
            except ValueError:
                return value

        return np.vectorize(try_number, otypes=[object])(data)

    @classmethod
    def __convert_birthday(cls, birthday_column) -> np.ndarray:
        actual_year = datetime.now().year
        birthday_column = [
            actual_year - datetime.strptime(date, "%Y-%m-%d").year
            for date in birthday_column                   
        ]

        return np.array(birthday_column)

    @classmethod
    def __convert_besthand(cls, besthand_column) -> np.ndarray:
        besthand_column[besthand_column == "Left"] = 0
        besthand_column[besthand_column == "Right"] = 1

        return besthand_column

    @classmethod
    def __extract_empty_values(cls, data) -> dict[int, list[int]]:
        """
        Return a list of empty values inside the columns
        - key correspond to the index of the column
        - list[int] correspond to each row index missing in the column

        Note:
        - column who are fully empty are ignored
        """
        missing = dict()

        for i, col in enumerate(data.T):
            indexes = np.argwhere((col == "")).flatten()

            if (len(indexes) > 0 and len(indexes) != len(col)):
                missing[i] = indexes
        return missing
