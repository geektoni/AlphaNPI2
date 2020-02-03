import pandas as pd
import glob
import os

import argparse
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default="PARTITION_UPDATE")
    args = parser.parse_args()

    result_files = glob.glob("../results/validation_*")
    operation_name = args.op

    # Possible value combinations
    combinations = [["True", "False", "False"],
                    ["True", "False", "True"],
                    ["False", "True", "False"],
                    ["False", "True", "True"],
                    ["False", "False", "False"],
                    ["False", "False", "True"]
                    ]

    combinations_labels = [
        "reduced (no stack)",
        "reduced (stack)",
        "without pd (no stack)",
        "without pd (stack)",
        "all (no stack)",
        "all (stack)"
    ]

    total_results = []

    # For each file, parse its content and generate a table
    for f in result_files:

        file_name = os.path.basename(f)

        file_name_values = file_name.split("-")
        operation = file_name_values[2]
        prob_samb_err = file_name_values[3]
        reduced_operation = file_name_values[4]
        without_partition = file_name_values[5]
        expose_stack = file_name_values[6]

        # File results
        results = [operation, prob_samb_err, reduced_operation, without_partition, expose_stack]

        with open(f, "r") as open_file:

            skip=0
            for line in open_file:
                if skip == 0:
                    skip += 1
                    continue

                # Split the various information
                # 0: length
                # 1: mcts mean
                # 2: mcts normalized
                # 3: net mean
                values = line.split(",")

                length = int(values[0].split(":")[1])
                mcts_norm = float(values[2].split(":")[1])
                net_mean = float(values[3].split(":")[1])

                # Appent the final result
                total_results.append(results+[length, mcts_norm, net_mean])

    # Generate the pandas dataframe
    df = pd.DataFrame(total_results, columns=["operation", "samp_err", "reduced", "no_part_upd", "expose_stack", "len", "mcts", "net"])
    df.sort_values(by=["operation", "len", "samp_err"], inplace=True)

    data_op = df[df.operation == operation_name]
    total_data_op = []
    total_data_op_2 = []
    # Length of the lists
    values_lengths = [5, 20, 60, 100]

    # Generate the list with all the data
    for c in combinations:
        list_tot = []
        list_tot_2 = []
        for v in values_lengths:
            with_without_train = data_op[(data_op.reduced == c[0])
                        & (data_op.no_part_upd == c[1])
                        & (data_op.expose_stack == c[2])
                        & (data_op.len == v)]["mcts"].to_list()
            if len(with_without_train) == 0:
                list_tot.append(0)
                list_tot_2.append(0)
                continue
            list_tot.append(with_without_train[0])
            list_tot_2.append(with_without_train[1])
        total_data_op_2.append(list_tot_2)
        total_data_op.append(list_tot)

    x = np.arange(len(values_lengths))  # the label locations
    width = 0.10  # the width of the bars

    fig, ax = plt.subplots(2)

    i=0
    rects = []
    for v in total_data_op:
        if i < 2:
            rect = ax[0].bar(x - width*(2-i) , v, width, label=combinations_labels[i])
        else:
            rect = ax[0].bar(x + width * (i-2), v, width, label=combinations_labels[i])
        i += 1
        rects.append((rect, ax[0]))

    i=0
    for v in total_data_op_2:
        if i < 2:
            rect = ax[1].bar(x - width* (2 - i), v, width, label=combinations_labels[i])
        else:
            rect = ax[1].bar(x + width * (i - 2), v, width, label=combinations_labels[i])
        i += 1
        rects.append((rect, ax[1]))

    ax[0].set_title("{} without error retrain".format(operation_name))
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(values_lengths)
    ax[1].set_title("{} with error retrain".format(operation_name))
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(values_lengths)

    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    for r in rects:
        autolabel(r[0], r[1])

    plt.tight_layout()
    plt.show()

    # Save the results to file
    df.to_csv("complete_results.csv")
