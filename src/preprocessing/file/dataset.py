import copy

class DataSet:
    def __init__(self, data: list[list], headers: dict):
        self._data = data
        self._headers = headers
        self._typed_data = self.to_float(data)
        self._columns = [[row[i] for row in self._typed_data] for i in range(len(self._typed_data[0]))]
        self._rows = [[c[i] for c in self._columns] for i in range(len(self._columns[0]))]

    def raw_data(self):
        return self._data

    def columns(self) -> list:
        return self._columns

    def column(self, i: int) -> list:
        return self._columns[i]

    def rows(self) -> list[list]:
        return self._rows

    def row(self, i: int) -> list:
        return self._rows[i]

    def name_column(self, name: str) -> list:
        return self.column(self._headers[name])

    def headers(self) -> list:
        return list(self._headers.keys())

    def to_float(self, data):
        cpy_data = copy.copy(data)

        for i in range(len(cpy_data)):
            for j in range(len(cpy_data[i])):
                try:
                    cpy_data[i][j] = float(cpy_data[i][j])
                except ValueError:
                    pass
        
        return cpy_data