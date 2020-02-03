import pandas as pd
import glob
import os


if __name__ == "__main__":

    result_files = glob.glob("../results/validation_*")

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

                len = int(values[0].split(":")[1])
                mcts_norm = float(values[2].split(":")[1])
                net_mean = float(values[3].split(":")[1])

                # Appent the final result
                total_results.append(results+[len, mcts_norm, net_mean])

    # Generate the pandas dataframe
    df = pd.DataFrame(total_results, columns=["operation", "samp_err", "reduced", "no_part_upd", "expose_stack", "len", "mcts", "net"])
    df.sort_values(by=["operation", "len", "samp_err"], inplace=True)
    print(df)

    # Save the results to file
    df.to_csv("complete_results.csv")
