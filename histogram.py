
from src.preprocessing.file.loader import CSVManager, DataSet
from src.preprocessing.maths import std
from matplotlib import pyplot as pp
import seaborn as sb


def plot(data: dict):
    for key, values in data.items():
        sb.histplot(values, bins=30, kde=True, label=key)
        break

    pp.legend()
    pp.show()

def main():
    dataset = CSVManager().load("datasets/dataset_train.csv")

    headers = dataset.headers()[6:]
    courses = dataset.columns()[6:]

    for k, v in enumerate(courses[0]):
        print(f"{k} : {v}")

    courses_std = [std(i) for i in courses]

    plot(dict(zip(headers, courses_std)))

if __name__ == "__main__":
    main()