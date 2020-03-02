import subprocess
import random

# set random seed
random.seed(42)

configs = ["complete", "without-partition-update", "reduced", "recursive"]
stack = [True, False]
train_errors = [True, False]
output_tb_dir = "./final_results_tb"

for c in configs:
    for s in stack:
        for t in train_errors:

            command = "bash submit_jobs.sh {} {} {} {} {}".format(
                c, s, t, output_tb_dir, random.randint(0, 10000)
            )

            # execute the command
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            print(output.decode('UTF-8'))