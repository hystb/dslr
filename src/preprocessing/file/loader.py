from src.preprocessing.maths import mean
from src.preprocessing.file.dataset import DataSet
from datetime import datetime
import numpy as np
import csv

class Loader:
    birtday_c_index = 4
    besthand_c_index = 5
    def load(self, filepath: str) -> DataSet:
        """
        Create a dataset object from a csv.file

        `data treatment`:
        - empty values of the dataset will be handled by `replacement` method
        - birthday values are converted to ages (years)
        - best hand is converted to int left/right -> 0/1
        - at the end values are standardized using z-score 
        """
        with open(filepath, 'r') as file:
            data = list(csv.reader(file, delimiter=","))

            headers = {k: v for v, k in enumerate(data.pop(0))}
            data_array = self.__to_matrix(data)

            data_treated = self.data_treatment(data_array.copy())

            print(f"original: {data_array}\ntreated: {data_treated}")

            return DataSet(data, headers)
        
    def data_treatment(self, data: np.ndarray) -> np.ndarray:
        # replace age and hands
        data[:, self.birtday_c_index] = self.__convert_birthday(data[:, self.birtday_c_index])
        data[:, self.besthand_c_index] = self.__convert_besthand(data[:, self.besthand_c_index])

        # numbers and empty values
        data = self.__convert_to_numbers(data)
        empty_dict = self.__extract_empty_values(data)

        self.replacement(empty_dict, data)

        return data

    def __to_matrix(self, data) -> np.ndarray:
        return np.array(data)
    
    def __convert_to_numbers(self, data) -> np.ndarray:
        def try_number(value):
            try:
                return float(value)
            except ValueError:
                return value

        return np.vectorize(try_number, otypes=[object])(data)

    def __convert_birthday(self, birthday_column) -> np.ndarray:
        actual_year = datetime.now().year
        birthday_column = [
            actual_year - datetime.strptime(date, "%Y-%m-%d").year
            for date in birthday_column                   
        ]

        return np.array(birthday_column)

    def __convert_besthand(self, besthand_column) -> np.ndarray:
        besthand_column[besthand_column == "Left"] = 0
        besthand_column[besthand_column == "Right"] = 1

        return besthand_column

    def __extract_empty_values(self, data) -> dict[int, list[int]]:
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

    def replacement(self, empty_values: dict, data) -> np.ndarray:
        result = data.copy()

        for key, value in empty_values.items():
            original_column: np.ndarray = data.T[key]
            original_without_missing = np.delete(original_column, value)

            for index in value:
                # replacement method
                result.T[key][index] = mean(original_without_missing)

        return result