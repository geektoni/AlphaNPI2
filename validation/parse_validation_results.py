import pandas as pd
import glob
import os
import itertools

import argparse
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.3)
sns.set_context("paper")
plt.rcParams.update({'font.size': 40})

def autolabel(rects, ax, ann_size):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=ann_size, fontweight="bold")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default="PARTITION_UPDATE")
    parser.add_argument("--dir", type=str, default="../results")
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--annotate", action="store_true", default=False)
    parser.add_argument("--title", action="store_true", default=False)
    parser.add_argument("--legend", action="store_true", default=False)
    parser.add_argument("--annotation-size", type=int, default=10)
    parser.add_argument("--line", action="store_true", default=False)
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--net", action="store_true", default=False)
    parser.add_argument("--latex", action="store_true", default=False)
    parser.add_argument("--std", action="store_true", default=False)
    args = parser.parse_args()

    result_files = glob.glob(args.dir+"/validation_*")
    operation_name = args.op

    # Possible value combinations
    # reduced, no_part_upd, expose_stack, recursive, expose_pointers
    combinations = [["True", "False", "False", "False", "False"],
                    ["False", "False", "False", "False", "False"],
                    ["True", "False", "False", "False", "True"],
                    ["False", "False", "False", "False", "True"],
                    ["False", "False", "False", "True", "False"]
                    ]

    combinations_labels = [
        "reduced (no pointers)",
        "all (no pointers)",
        "reduced (pointers)",
        "all (pointers)",
        "recursive (no pointers)"
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
        recursive_quicksort = file_name_values[7]
        do_not_expose_pointers = file_name_values[8]

        # File results
        results = [operation, prob_samb_err, reduced_operation, without_partition, expose_stack,
                   recursive_quicksort, do_not_expose_pointers]

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
                mcts_std = float(values[4].split(":")[1])
                net_std = float(values[5].split(":")[1].replace("\n", ""))

                # Appent the final result
                total_results.append(results+[length, mcts_norm, net_mean, mcts_std, net_std])

    # Generate the pandas dataframe
    df = pd.DataFrame(total_results, columns=["operation", "samp_err", "reduced", "no_part_upd", "expose_stack", "recursive", "expose_pointers", "len", "mcts", "net", "mcts_std", "net_std"])
    df.sort_values(by=["operation", "len", "samp_err"], inplace=True)

    data_op = df[df.operation == operation_name]
    total_data_op = []
    total_data_op_2 = []
    # Length of the lists
    values_lengths = np.arange(5, 65, 5)

    # Choose which measure whats to read (mcts or net)
    method = "mcts" if not args.net else "net"

    # Generate the list with all the data
    for c in combinations:
        list_tot = []
        list_tot_2 = []
        for v in values_lengths:
            with_without_train = data_op[(data_op.reduced == c[0])
                        & (data_op.no_part_upd == c[1])
                        & (data_op.expose_stack == c[2])
                        & (data_op.len == v)
                        & (data_op.recursive == c[3])
                        & (data_op.expose_pointers == c[4])][method].to_list()
            if len(with_without_train) == 0:
                list_tot.append(0)
                list_tot_2.append(0)
                continue
            if len(with_without_train) == 1:
                list_tot.append(with_without_train[0])
                list_tot_2.append(0)
                continue
            list_tot.append(with_without_train[0])
            list_tot_2.append(with_without_train[1])
        total_data_op_2.append(list_tot_2)
        total_data_op.append(list_tot)

    x = np.arange(len(values_lengths))  # the label locations
    width = 0.30  # the width of the bars
    markers_list = itertools.cycle(('.', '^', 'v', 's', 'P', "*", "D"))

    if len(total_data_op) != 1 and len(total_data_op_2) != 1:
        if args.line:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        else:
            fig, ax = plt.subplots(2, figsize=(10, 4))

    i=0
    rects = []
    if len(total_data_op) == 1 and len(total_data_op_2) == 1:
        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax = [ax, ax]

        if args.line:
            rect = ax[0].plot(total_data_op[0], label="Stardard Training")
            rect_2 = ax[0].plot(total_data_op_2[0], label="Re-train over ")
        else:
            rect_2 = ax[0].bar(x - width / 2, total_data_op_2[0], width, label="Re-train with failed sequences")
            rect = ax[0].bar(x+width/2, total_data_op[0], width, label="Default Training")

        rects.append((rect, ax[0]))
        rects.append((rect_2, ax[0]))
    else:
        for v in total_data_op:
            if i < 2:
                if args.line:
                    rect = ax[0].plot(v, label=combinations_labels[i], marker=next(markers_list))
                else:
                    rect = ax[0].bar(x - width*(2-i) , v, width, label=combinations_labels[i])
            else:
                if args.line:
                    rect = ax[0].plot(v, label=combinations_labels[i], marker=next(markers_list))
                else:
                    rect = ax[0].bar(x + width * (i-2), v, width, label=combinations_labels[i])
            i += 1
            rects.append((rect, ax[0]))

    # Regenerate the iterator since we cannot rewind it
    markers_list = itertools.cycle(('.', '^', 'v', 's', 'P', "*", "D"))
    i=0
    if len(total_data_op_2) != 1:
        for v in total_data_op_2:
            if i < 2:
                if args.line:
                    rect = ax[1].plot(v, label=combinations_labels[i], marker=next(markers_list))
                else:
                    rect = ax[1].bar(x - width* (2 - i), v, width, label=combinations_labels[i])
            else:
                if args.line:
                    rect = ax[1].plot(v, label=combinations_labels[i], marker=next(markers_list))
                else:
                    rect = ax[1].bar(x + width * (i - 2), v, width, label=combinations_labels[i])
            i += 1
            rects.append((rect, ax[1]))

    if args.title:
        ax[0].set_title("{} without error retrain".format(operation_name))
        ax[1].set_title("{} with error retrain".format(operation_name))
        if args.line:
            ax[0].set_title("Without error sampling")
            ax[1].set_title("With error sampling")


    ax[0].set_xticks(x)
    ax[0].set_xticklabels(values_lengths)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(values_lengths)

    # Set plot limits
    ax[0].set_ylim(0,1)
    ax[1].set_ylim(0,1)


    if args.legend:
        if not args.line:
            box = ax[0].get_position()
            ax[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        box = ax[1].get_position()
        ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #title=operation_name
        ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title_fontsize=10)

    if args.annotate:
        for r in rects:
            autolabel(r[0], r[1], args.annotation_size)

    plt.tight_layout()

    plt.ylim(0, 1)

    if not args.save:
        if args.show:
            plt.show()
    else:
        if args.net:
            name = "net"
        else:
            name = "mcts"

        plt.savefig("{}_plot_{}.png".format(args.op, name), dpi=300, bbox_inches="tight")

    # Save the results to file
    df.to_csv("complete_results.csv")

    # generate latex output
    if args.latex:

        combinations = [["False", "False"],
                        ["True", "False"],
                        ["False", "True"]
                        ]

        model = "mcts" if not args.net else "net"
        std_name = "mcts_std" if not args.net else "net_std"
        with open("output_latex_{}.txt".format(model), "w+") as output_latex:
            method=args.op
            for v in [5, 10, 20, 50]:
                output_latex.write("{} ".format(v))
                for c in combinations:
                    latex_data = df[(df.reduced == c[0])
                        & (df.no_part_upd == c[1])
                        & (df.len == v)
                        & (df.operation == method)]

                    if len(latex_data) == 4:

                        no_samp_no_stack = latex_data[(latex_data.samp_err == "0.0")
                               & (latex_data.expose_stack == "False")][[model, std_name]].values[0]
                        samp_no_stack = latex_data[(latex_data.samp_err == "0.3")
                               & (latex_data.expose_stack == "False")][[model, std_name]].values[0]
                        no_samp_stack = latex_data[(latex_data.samp_err == "0.0")
                               & (latex_data.expose_stack == "True")][[model, std_name]].values[0]
                        samp_stack = latex_data[(latex_data.samp_err == "0.3")
                               & (latex_data.expose_stack == "True")][[model, std_name]].values[0]

                        values = [no_samp_no_stack[0],
                        samp_no_stack[0],
                        no_samp_stack[0],
                        samp_stack[0]]

                        values_printable = [
                        #"{:.2f}".format(no_samp_no_stack[0]),
                        "{:.2f}".format(samp_no_stack[0]),
                        #"{:.2f}".format(no_samp_stack[0]),
                        "{:.2f}".format(samp_stack[0])
                        ]

                        values_std = [
                            #"{:.2f}".format(no_samp_no_stack[1]),
                            "{:.2f}".format(samp_no_stack[1]),
                            #"{:.2f}".format(no_samp_stack[1]),
                            "{:.2f}".format(samp_stack[1])
                        ]

                        max_value = max(values_printable)

                        for i in range(0, len(values_printable)):
                            if args.std:
                                if values_printable[i] == max_value and values_printable[i] != "0.00":
                                    values_printable[i] = "\\textbf{" + values_printable[i] + "$\pm$" + values_std[i] + "}"
                                else:
                                    values_printable[i] = values_printable[i] + "$\pm$" + values_std[i]
                            else:
                                if values_printable[i] == max_value and values_printable[i] != "0.00":
                                    values_printable[i] = "\\textbf{"+values_printable[i]+"}"
                                else:
                                    values_printable[i] = values_printable[i]

                        output_latex.write(
                                "& {} & {}".format(
                                    values_printable[0],
                                    values_printable[1],
                                    #values_printable[2],
                                    #values_printable[3]
                                )
                        )
                    else:
                        output_latex.write("& None & None")

                output_latex.write(" \\\\ \\hline \n")
