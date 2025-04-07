from src.preprocessing.maths import *

class Z_score():
    def normalize_list(self, data):
        mean = sum(data) / len(data)
        standard_deviation = math.sqrt(sum((data - mean)**2) / len(data))
        return [self.__normalize(x, mean, standard_deviation).item() for x in data]
    
    def __normalize(self, x, mean, standard_deviation):
        return (x - mean) / standard_deviation