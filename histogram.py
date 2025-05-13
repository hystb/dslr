import os
import sys
sys.path.append('./src/preprocessing/file')

import matplotlib.pyplot as plt
import numpy as np
from dataset import DataSet
from loader import Loader


def do_histplot(dataset: DataSet):
    """
    Function to plot a histogram of the scores for each house in a given course.
    """
    colors = {
        "Gryffindor": "red",
        "Ravenclaw": "blue",
        "Hufflepuff": "gold",
        "Slytherin": "green"
    }
    loader = Loader()
    dataset = loader.load("./datasets/dataset_train.csv")

    data = dataset.get_all()
    headers = dataset.get_headers()

    course = "Care of Magical Creatures"
    course_idx = headers[course]
    houses_col_idx = headers["Hogwarts House"]
    unique_houses = np.unique(data[:, houses_col_idx])
    fig, ax = plt.subplots(figsize=(8, 6))

    for house in unique_houses:
        mask = data[:, houses_col_idx] == house
        scores = data[mask, course_idx]
        scores = scores[scores != ''] 
        scores = scores.astype(float)
        ax.hist(scores, bins=15, alpha=0.5, label=house, color=colors[house])
    ax.set_title(course)
    ax.set_xlabel("Score")
    ax.set_ylabel("Nb of students")
    ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    if (len(sys.argv) == 2 and os.path.exists(sys.argv[1])):
        dataset = Loader().load(sys.argv[1])
        do_histplot(dataset)
    else:
        print("usage: python histogram.py [dataset]")

if __name__ == "__main__":
    main()
