import argparse
import pandas as pd



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="complete_results.csv")
    parser.add_argument("--net", action="store_true", default=False)
    parser.add_argument("--stack", action="store_true", default=False)
    parser.add_argument("--std", action="store_true", default=False)

    args = parser.parse_args()

    # Read the dataframe
    df = pd.read_csv(args.file)

    # Possible combinations
    combinations = [[False, False],
                    [True, False],
                    [False, True]
                    ]

    # All the operations
    operations = ["PARTITION_UPDATE", "PARTITION", "SAVE_LOAD_PARTITION", "QUICKSORT_UPDATE", "QUICKSORT"]

    # Target length
    v=20

    first_type = "Stack" if args.stack else "$s=0.3$"
    second_type = "No Stack" if args.stack else "$s=0.0$"

    model = "mcts" if not args.net else "net"
    std_name = "mcts_std" if not args.net else "net_std"
    with open("output_latex_{}.txt".format(model), "w+") as output_latex:

        output_latex.write("""
\\begin{table}[]
\centering
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{|l!{\\vrule width 1.6pt}l|l!{\\vrule width 1.6pt}l|l!{\\vrule width 1.6pt}l|l!{\\vrule width 1.6pt}}
\hline
\multicolumn{1}{|c|}{\\textbf{Operation}} & \multicolumn{2}{c|}{$M_{all}$} & \multicolumn{2}{c|}{$M_{red}$} & \multicolumn{2}{c|}{$M_{wupd}$} \\\\ \cline{2-7}
""")

        output_latex.write(
            "\multicolumn{1}{|c|}{} & \multicolumn{1}{c|}{\\textbf{"+second_type+"}} & \multicolumn{1}{c|}{\\textbf{"+first_type+"}} & \multicolumn{1}{c|}{\\textbf{"+second_type+"}} & \multicolumn{1}{c|}{\\textbf{"+first_type+"}} & \multicolumn{1}{c|}{\\textbf{"+second_type+"}} & \multicolumn{1}{c|}{\\textbf{"+first_type+"}} \\\ \hline \n")

        for method in operations:
            output_latex.write("\\texttt{{ {} }} ".format(method.lower().replace("_", "\_")))
            for c in combinations:
                latex_data = df[(df.reduced == c[0])
                                & (df.no_part_upd == c[1])
                                & (df.len == v)
                                & (df.operation == method)]

                if len(latex_data) == 4:

                    if args.stack:
                        no_sampling = latex_data[(latex_data.samp_err == 0.3)
                                                 & (latex_data.expose_stack == False)][[model, std_name]].values[0]
                        sampling = latex_data[(latex_data.samp_err == 0.3)
                                              & (latex_data.expose_stack == True)][[model, std_name]].values[0]
                    else:
                        no_sampling=latex_data[(latex_data.samp_err == 0.0)
                                       & (latex_data.expose_stack == False)][[model, std_name]].values[0]
                        sampling=latex_data[(latex_data.samp_err == 0.3)
                                       & (latex_data.expose_stack == False)][[model, std_name]].values[0]

                    if args.std:
                        no_sampling_str = "\\textbf{{ {:.2f}$\pm${:.2f} }}".format(no_sampling[0], no_sampling[1]) if no_sampling[0] >= sampling[0] and no_sampling[0] > 1**10-15  else "{:.2f}$\pm${:.2f}".format(no_sampling[0], no_sampling[1])
                        sampling_str = "\\textbf{{ {:.2f}$\pm${:.2f} }}".format(sampling[0], sampling[1]) if no_sampling[0] <= sampling[0] and no_sampling[0] > 1**10-15 else "{:.2f}$\pm${:.2f}".format(sampling[0], sampling[1])
                    else:
                        no_sampling_str = "\\textbf{{ {:.2f} }}".format(no_sampling[0]) if no_sampling[0] >= sampling[0] and no_sampling[0] > 1**10-15 else "{:.2f}".format(no_sampling[0])
                        sampling_str = "\\textbf{{ {:.2f} }}".format(sampling[0]) if no_sampling[0] <= sampling[0] and sampling[0] > 1**10-15 else "{:.2f}".format(sampling[0])


                    output_latex.write(
                        "& {} &  {}  ".format(
                            no_sampling_str, sampling_str
                        )
                    )
                else:
                    output_latex.write("& None & None")
            output_latex.write(" \\\\ \\hline \n")

        output_latex.write("""
\end{tabular}%
}
\caption{}
\label{tab:result_table}
\end{table}
 """)
