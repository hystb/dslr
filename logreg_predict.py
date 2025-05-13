import os
import sys
import csv
from src.preprocessing.file.dataset import DataSet
from src.processing.model_manager import ModelManager
from src.preprocessing.file.replacer import MeanReplacer
from src.preprocessing.file.loader import Loader

to_exclude = ["Birthday", "Best Hand", "Arithmancy", "Potions", "Care of Magical Creatures"]

def export_to_csv(file, data):
    with open(file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Index", "Hogwarts House"])
        writer.writeheader()
        writer.writerows(data)

def do_predict(dataset: DataSet, model_file: str):
    data = dataset.standardize()

    core = ModelManager().import_from_file(model_file)
    X = data.get_numerics(exclude=to_exclude)

    predictions = [core.predict(t) for t in X]
    data = [{"Index":i, "Hogwarts House":k} for i, k in enumerate(predictions)]
    export_to_csv("houses.csv", data)
    print("Result successfully exported to houses.csv")

def main():
    if (len(sys.argv) == 3 and os.path.exists(sys.argv[1]) and os.path.exists(sys.argv[2])):
        dataset = Loader().load(sys.argv[1], replacer=MeanReplacer)
        do_predict(dataset, sys.argv[2])
    else:
        print("usage: python logreg_predict.py [dataset] [model]")

if __name__ == "__main__":
    main()
