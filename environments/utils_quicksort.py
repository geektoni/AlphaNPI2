import numpy as np

def assert_partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp):
    assert init_pointers_pos3 <= init_pointers_pos2 \
           and init_pointers_pos1 <= init_pointers_pos2 \
           and init_pointers_pos1 <= init_pointers_pos3 and temp[0] != -1, "Partition update {}, {}, {}, {}".format(
        init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, temp)

def assert_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp):
    assert init_pointers_pos3 == init_pointers_pos1 \
           and init_pointers_pos1 <= init_pointers_pos2 \
           and init_pointers_pos1 <= init_pointers_pos3 and temp[0] == init_pointers_pos1, "Partition {}, {}, {}, {}".format(
        init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, temp)

def assert_quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp):
    assert len(stack) >= 3, "Quicksort Update: {}".format(stack)


def should_stop(length):
    return np.random.choice(2, 1, p=[1 - (1 / length), 1 / length])[0] == 1

def partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, stop, stop_partition_update=False):

    if should_stop(len(scratchpad_ints)) and stop_partition_update:
        stop = True
        assert_partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

    if scratchpad_ints[init_pointers_pos3] <= scratchpad_ints[init_pointers_pos2] and not stop:
        scratchpad_ints[[init_pointers_pos3, init_pointers_pos1]] = scratchpad_ints[
            [init_pointers_pos1, init_pointers_pos3]]
        init_pointers_pos1 += 1
    init_pointers_pos3 += 1

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, stop


def partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, stop, stop_partition=False, stop_partition_update=False):

    if np.random.choice(2, 1, p=[1 - (1 / len(scratchpad_ints)), 1 / len(scratchpad_ints)])[0] == 1 and stop_partition:
        stop = True
        assert_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

    while init_pointers_pos3 < init_pointers_pos2 and not stop:
        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, stop = \
        partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, stop, stop_partition_update)

    if not stop:
        scratchpad_ints[[init_pointers_pos1, init_pointers_pos2]] = scratchpad_ints[
            [init_pointers_pos2, init_pointers_pos1]]

    return scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, stop


def quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, stop, stop_partition=False, stop_partition_update=False, stop_quicksort_update=False):

    if should_stop(len(scratchpad_ints)) and stop_quicksort_update:
        stop = True

    if not stop:
        init_pointers_pos1 = init_prog_stack.pop()
        init_pointers_pos2 = init_prog_stack.pop()
        init_pointers_pos3 = init_prog_stack.pop()

    if init_pointers_pos1 < init_pointers_pos2 and not stop:

        init_temp_variables = [init_pointers_pos1]

        # Run the partition method
        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, \
        init_pointers_pos3, init_prog_stack, init_temp_variables, stop = \
            partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, stop, stop_partition, stop_partition_update)

        if init_pointers_pos1 + 1 < init_pointers_pos2 and not stop:
            init_prog_stack.append(init_pointers_pos1 + 1)
            init_prog_stack.append(init_pointers_pos2)
            init_prog_stack.append(init_pointers_pos1 + 1)

        if init_pointers_pos1 - 1 > 0 and not stop:
            if init_temp_variables[0] < init_pointers_pos1 - 1:
                init_pointers_pos3 = init_temp_variables[0]
                init_prog_stack.append(init_pointers_pos3)
                init_prog_stack.append(init_pointers_pos1 - 1)
                init_prog_stack.append(init_pointers_pos3)

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy(), stop

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
        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, stop = \
           quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, stop, stop_partition, stop_partition_update, stop_quicksort_update)

    # This means that we reached the end of the sorting without
    # exiting the loop. Therefore, we initialize everything manually.
    if not stop and not sort:
        np.random.shuffle(scratchpad_ints)
        init_pointers_pos1 = 0
        init_pointers_pos2 = length - 1
        init_pointers_pos3 = 0
        if (stop_quicksort_update):
            init_prog_stack = [0, length-1, 0]
        else:
            init_prog_stack = []
        init_temp_variables = [init_pointers_pos1]

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy()


# Testing
if __name__ == "__main__":
    for i in range(0,10000):
        arr = np.random.randint(0, 100, 5)
        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = \
            sample_quicksort_indexes(np.copy(arr), 5, sort=True)

        #assert_partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)
        #assert_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)
        #assert_quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

        if not np.array_equal(np.array(sorted(arr)), np.array(scratchpad_ints)):
            print("{}, {}".format(np.array(sorted(arr)), np.array(scratchpad_ints)))