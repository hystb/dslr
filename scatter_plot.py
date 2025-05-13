import os
import sys
sys.path.append('./src/preprocessing/file')

import matplotlib.pyplot as plt
import numpy as np
from dataset import DataSet
from loader import Loader


def do_scatter(dataset: DataSet):
    loader = Loader()
    dataset = loader.load("./datasets/dataset_train.csv")
    data = dataset.get_all()
    headers = dataset.get_headers()

    x_course = "Astronomy"
    y_course = "Defense Against the Dark Arts"

    idx_x = headers[x_course]
    idx_y = headers[y_course]
    idx_house = headers["Hogwarts House"]

    colors = {
        "Gryffindor": "red",
        "Ravenclaw": "blue",
        "Hufflepuff": "gold",
        "Slytherin": "green"
    }

    plt.figure(figsize=(6, 6))
    for house, color in colors.items():
        mask_house = data[:, idx_house] == house
        x_vals = data[mask_house, idx_x]
        y_vals = data[mask_house, idx_y]

        mask_valid = (x_vals != "") & (y_vals != "")
        x_vals = x_vals[mask_valid].astype(float)
        y_vals = y_vals[mask_valid].astype(float)

        plt.scatter(x_vals, y_vals, alpha=0.5, label=house, color=color, s=10)

    plt.xlabel(x_course)
    plt.ylabel(y_course)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    if (len(sys.argv) == 2 and os.path.exists(sys.argv[1])):
        dataset = Loader().load(sys.argv[1])
        do_scatter(dataset)
    else:
        print("usage: python scatter_plot.py [dataset]")

if __name__ == "__main__":
    main()
