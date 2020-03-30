from tensorboard.backend.event_processing import event_accumulator

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.3)
plt.rcParams.update({'font.size': 40})

import argparse

import glob

import os

import pandas as pd

if __name__ == "__main__":

    parser = argparse.ArgumentParser("")
    parser.add_argument("--dir", type=str, default="./tb")

    args = parser.parse_args()

    result_files = glob.glob(args.dir + "/*.csv")

    total_data = pd.DataFrame()
    counter=0
    for f in result_files:

        file_name = os.path.basename(f)

        file_name_values = file_name.split("-")
        penalty = file_name_values[5].lower() == "true"

        if penalty:
            value = "Original MCTS"
        else:
            value = "Modified MCTS"

        data = pd.read_csv(f).drop(["Wall time"], axis=1)
        data.rename(columns={"Step": "Step_"+str(counter), "Value": value}, inplace=True)
        total_data =  pd.concat([total_data, data], axis=1)
        counter+=1

    total_data.dropna(inplace=True)
    total_data.drop(["Step_"+str(counter-1)], axis = 1, inplace=True)
    total_data.rename(columns={"Step_0": "Step"}, inplace=True)
    total_data.drop(["Step"], axis=1, inplace=True)

    fig, ax = plt.subplots(1,1, figsize=(10, 5))

    #sns.lineplot(data=total_data, dashes=False, linewidth=3)
    total_values = len(total_data["Original MCTS"])
    plt.plot(range(total_values), total_data["Modified MCTS"], linewidth=3, label="Modified MCTS")
    plt.plot(range(total_values), total_data["Original MCTS"], linewidth=3, label="Original MCTS")
    plt.plot(range(-100, total_values+100), [1 for _ in range(-100,total_values+100)], color='k', linestyle='--', linewidth=2)

    plt.text(60, 1.05, "Target Program Accuracy (1.0)", fontsize=10, color="red", fontweight="bold")

    plt.legend(fontsize=10)

    plt.xlabel("Training Epochs", fontweight="bold")
    plt.ylabel("Training Accuracy", fontweight="bold")

    plt.xlim([0, total_values])
    plt.ylim([0, 1.2])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig("l_penalty.png", dpi=300, bbox_inches="tight")