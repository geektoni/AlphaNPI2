import subprocess
import glob

output_dir = "../models"

for f in glob.glob(output_dir+"/*.pth"):

    # split filename
    values = f.split("-")

    # Get the values we need
    expose_stack = values[6] == "True"
    without_partition = values[8] == "True"
    reduced = values[9] == "True"

    # Get correct operations
    operations="none"
    if reduced:
        operations="reduced"
    elif without_partition:
        operations="no-partition"

    # Generate the command
    command = "bash submit_validate.sh {} {}".format(
        f, operations
        )
    print(command)

    # execute the command
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)