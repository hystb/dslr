import os
import sys
from sklearn.metrics import accuracy_score
from src.preprocessing.file.dataset import DataSet
from src.processing.model_manager import ModelManager
from src.preprocessing.file.replacer import MeanReplacer
from src.preprocessing.file.loader import Loader
from src.processing.logistic_regression import OneVsAllLogisticRegression

to_exclude = ["Birthday", "Best Hand", "Arithmancy", "Potions", "Care of Magical Creatures"]

def do_train(dataset: DataSet):
    data = dataset.standardize()

    X = data.get_numerics(exclude=to_exclude)
    y = data.get_column("Hogwarts House")

    core = OneVsAllLogisticRegression()
    core.fit(X, y)

    predictions = [core.predict(t) for t in X]
    print(f"OneVsAll model accuracy: {accuracy_score(predictions, y):.3f}")

    ModelManager().export_to_file(core, "dslr.model")
    print("Model successfully exported!")

def main():
    if (len(sys.argv) == 2 and os.path.exists(sys.argv[1])):
        dataset = Loader().load(sys.argv[1], replacer=MeanReplacer)
        do_train(dataset)
    else:
        print("usage: python logreg_train.py [dataset]")

if __name__ == "__main__":
    main()