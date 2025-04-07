from src.preprocessing.file.dataset import DataSet
import csv

class CSVManager:
    def load(self, filepath: str) -> DataSet:
        with open(filepath, 'r') as file:
            data = list(csv.reader(file, delimiter=","))

            headers = {k: v for v, k in enumerate(data.pop(0))}
            return DataSet(data, headers)
