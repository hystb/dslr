import sys
sys.path.append('./src/preprocessing/file')

import matplotlib.pyplot as plt
import numpy as np
from dataset import DataSet
from loader import Loader


def do_pair(dataset: DataSet):
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

    excluded = {"Index", "Hogwarts House", "First Name", "Last Name", "Birthday"}
    course_cols = [k for k in headers if k not in excluded]
    n = len(course_cols)
    houses_col_idx = headers["Hogwarts House"]

    fig, axs = plt.subplots(n, n, figsize=(n * 2.5, n * 2.5))

    for i, row_course in enumerate(course_cols):
        for j, col_course in enumerate(course_cols):
            ax = axs[i, j]
            idx_row = headers[row_course]
            idx_col = headers[col_course]

            for house, color in colors.items():
                mask_house = (data[:, houses_col_idx] == house)
                x = data[:, idx_col]
                y = data[:, idx_row]

                mask_valid = (x != "") & (y != "") & mask_house
                x_vals = x[mask_valid].astype(float)
                y_vals = y[mask_valid].astype(float)

                ax.scatter(x_vals, y_vals, alpha=0.4, label=house, color=color, s=5)

            if i == n - 1:
                ax.set_xlabel(col_course, fontsize=7)
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(row_course, fontsize=7)
            else:
                ax.set_yticks([])

            ax.tick_params(axis='both', which='major', labelsize=6)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 1))

    plt.tight_layout()
    plt.show()


def main():
    if (len(sys.argv) == 2):
        dataset = Loader().load(sys.argv[1])
        do_pair(dataset)


if __name__ == "__main__":
    main()
