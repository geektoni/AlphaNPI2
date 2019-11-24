import numpy as np


def sample_quicksort_indexes(scratchpad_ints, length, sort=False, stop_partition=False, stop_partition_update=False, stop_quicksort_update=False):

    init_pointers_pos1 = 0
    init_pointers_pos2 = length - 1
    init_pointers_pos3 = 0

    init_temp_variables = [-1]

    init_prog_stack = []
    init_prog_stack.append(init_pointers_pos3)
    init_prog_stack.append(init_pointers_pos2)
    init_prog_stack.append(init_pointers_pos1)

    stop = False
    while len(init_prog_stack) > 0 and not stop:
        # Execute one round of quicksort
        init_pointers_pos1 = init_prog_stack.pop()
        init_pointers_pos2 = init_prog_stack.pop()
        init_pointers_pos3 = init_prog_stack.pop()

        if init_pointers_pos1 < init_pointers_pos2:

            init_temp_variables = [init_pointers_pos1]

            if np.random.choice(2,1, p=[1-(1/length), 1/length])[0] == 1 and stop_partition:
                stop = True
                break

            # PARTITION FUNCTION
            while init_pointers_pos3 < init_pointers_pos2 and not stop:
                if scratchpad_ints[init_pointers_pos3] <= scratchpad_ints[init_pointers_pos2]:
                    scratchpad_ints[[init_pointers_pos3, init_pointers_pos1]] = scratchpad_ints[
                        [init_pointers_pos1, init_pointers_pos3]]
                    init_pointers_pos1 += 1
                init_pointers_pos3 += 1

                if np.random.choice(2,1, p=[1-(1/length), 1/length])[0] == 1 and stop_partition_update:
                    stop = True
                    break

            if not stop:
                scratchpad_ints[[init_pointers_pos1, init_pointers_pos2]] = scratchpad_ints[[init_pointers_pos2, init_pointers_pos1]]

            if init_pointers_pos1 + 1 < init_pointers_pos2 and not stop:
                init_prog_stack.append(init_pointers_pos1 + 1)
                init_prog_stack.append(init_pointers_pos2)
                init_prog_stack.append(init_pointers_pos1 + 1)

            if init_pointers_pos1 - 1 > 0 and not stop:
                if init_temp_variables[0] < init_pointers_pos1-1:
                    init_pointers_pos3 = init_temp_variables[0]
                    init_prog_stack.append(init_pointers_pos3)
                    init_prog_stack.append(init_pointers_pos1 - 1)
                    init_prog_stack.append(init_pointers_pos3)

            if np.random.choice(2,1, p=[1-(1/length), 1/length])[0] == 1 and stop_quicksort_update:
                stop = True
                break

    # This means that we reached the end of the sorting without
    # exiting the loop. Therefore, we initialize everything manually.
    if not stop and not sort:
        np.random.shuffle(scratchpad_ints)
        init_pointers_pos1 = 0
        init_pointers_pos2 = length - 1
        init_pointers_pos3 = 0
        init_prog_stack = []
        init_temp_variables = [init_pointers_pos1]

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy()


# Testing
if __name__ == "__main__":
    for i in range(0,1000):
        arr = np.random.randint(0, 10, 10)
        out, _, _, _, _, _ = sample_quicksort_indexes(np.copy(arr), 10, sort=True)
        if not np.array_equal(np.array(sorted(arr)), np.array(out)):
            print("{}, {}".format(np.array(sorted(arr)), np.array(out)))