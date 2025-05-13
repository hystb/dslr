from src.processing.logistic_regression import OneVsAllLogisticRegression
import pickle

class ModelManager():
    @staticmethod
    def export_to_file(system: OneVsAllLogisticRegression, filepath: str):
        with open(filepath, "wb") as file:
            pickle.dump(system.models, file)

    @staticmethod
    def import_from_file(filepath: str) -> OneVsAllLogisticRegression:
        system = OneVsAllLogisticRegression()

        with open(filepath, "rb") as file:
            system.models = pickle.load(file)
        return system
