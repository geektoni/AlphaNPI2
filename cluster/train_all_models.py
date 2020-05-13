import subprocess
import random

# set random seed
random.seed(42)
seed = random.randint(0, 10000)

configs = ["complete", "without-partition-update", "reduced", "recursive"]
stack = [True, False]
train_errors = [0.3, 0.0]
expose_pointers = [True, False]
output_tb_dir = "./final_results_tb"

for c in configs:
    for s in stack:
        for t in train_errors:
            for exp in expose_pointers:

                command = "bash submit_jobs.sh {} {} {} {} {} {}".format(
                    c, s, t, output_tb_dir, seed, exp
                )

                # execute the command
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()
                print(output.decode('UTF-8'))
