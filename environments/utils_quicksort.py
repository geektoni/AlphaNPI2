import numpy as np

def assert_partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp):
    assert init_pointers_pos3 < init_pointers_pos2 \
           and init_pointers_pos1 < init_pointers_pos2 \
           and init_pointers_pos1 <= init_pointers_pos3 and temp[0] != -1, "Partition update {}, {}, {}, {}".format(
        init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, temp)

def assert_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp):
    assert init_pointers_pos3 == init_pointers_pos1 \
           and init_pointers_pos1 < init_pointers_pos2 \
           and temp[0] == init_pointers_pos1, "Partition {}, {}, {}, {}".format(
        init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, temp)

def asser_save_load_partition(init_pointers_pos1, init_pointers_pos2):
    assert init_pointers_pos1 < init_pointers_pos2 and init_pointers_pos1 == init_pointers_pos3, "Save Load Partition {}, {}".format(init_pointers_pos1, init_pointers_pos2)

def random_push(init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_temp_variables, init_prog_stack, stop):

    val = np.random.randint(0,2)

    if val == 1:
        if init_pointers_pos1 - 1  > 0 and init_temp_variables[0] < init_pointers_pos1 - 1 and not stop:
            init_prog_stack.append(init_pointers_pos3)
            init_prog_stack.append(init_pointers_pos1 - 1)
            init_prog_stack.append(init_pointers_pos3)

        if init_pointers_pos1 + 1 < init_pointers_pos2 and not stop:
            init_prog_stack.append(init_pointers_pos1 + 1)
            init_prog_stack.append(init_pointers_pos2)
            init_prog_stack.append(init_pointers_pos1 + 1)
    else:

        if init_pointers_pos1 + 1 < init_pointers_pos2 and not stop:
            init_prog_stack.append(init_pointers_pos1 + 1)
            init_prog_stack.append(init_pointers_pos2)
            init_prog_stack.append(init_pointers_pos1 + 1)

        if init_pointers_pos1 - 1 > 0 and init_temp_variables[0] < init_pointers_pos1 - 1 and not stop:
            init_prog_stack.append(init_pointers_pos3)
            init_prog_stack.append(init_pointers_pos1 - 1)
            init_prog_stack.append(init_pointers_pos3)

    return init_prog_stack.copy()


def assert_quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp):
    assert len(stack) >= 3, "Quicksort Update: {}".format(stack)

def partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, stop, stop_partition_update=False):

    """ (3)
    Representation as sub commands:
    * SWAP_PIVOT
    * PTR_1_RIGHT
    * STOP

    :param scratchpad_ints:
    :param init_pointers_pos1:
    :param init_pointers_pos2:
    :param init_pointers_pos3:
    :param stack:
    :param temp:
    :param stop:
    :param stop_partition_update:
    :return:
    """

    if np.random.choice(2, 1, p=[0.7, 0.3])[0] == 1 and stop_partition_update:
        stop = True
        assert_partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

    if scratchpad_ints[init_pointers_pos3] < scratchpad_ints[init_pointers_pos2] and not stop:
        scratchpad_ints[[init_pointers_pos3, init_pointers_pos1]] = scratchpad_ints[
            [init_pointers_pos1, init_pointers_pos3]]
        init_pointers_pos1 += 1

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, stop


def partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, stop, stop_partition=False, stop_partition_update=False):
    """
    (total of 2*(n-1)+2)
    from 0 to n-1:
        PARTITION_UPDATE
        PTR_3_RIGHT
    SWAP
    STOP

    :param scratchpad_ints:
    :param init_pointers_pos1:
    :param init_pointers_pos2:
    :param init_pointers_pos3:
    :param stack:
    :param temp:
    :param stop:
    :param stop_partition:
    :param stop_partition_update:
    :return:
    """

    if np.random.choice(2, 1, p=[0.5, 0.5])[0] == 1 and stop_partition:
        stop = True
        assert_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

    while init_pointers_pos3 < init_pointers_pos2 and not stop:
        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, stop = \
        partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, stop, stop_partition_update)
        if not stop:
            init_pointers_pos3 += 1

    if not stop:
        scratchpad_ints[[init_pointers_pos1, init_pointers_pos2]] = scratchpad_ints[
            [init_pointers_pos2, init_pointers_pos1]]

    return scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, stop

def save_load_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, stop, stop_partition=False, stop_partition_update=False, stop_save_load_partition=False):

    """ (4 operations)
    SAVE_PTR1
    PARTITION
    LOAD_PTR1
    STOP

    :param scratchpad_ints:
    :param init_pointers_pos1:
    :param init_pointers_pos2:
    :param init_pointers_pos3:
    :param init_prog_stack:
    :param init_temp_variables:
    :param stop:
    :param stop_partition:
    :param stop_partition_update:
    :param stop_save_load_partition:
    :return:
    """

    if np.random.choice(2, 1, p=[0.5, 0.5])[0] == 1 and stop_save_load_partition:
        stop = True

    if not stop:
        init_temp_variables = [init_pointers_pos1]

        # Run the partition method
        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, \
        init_pointers_pos3, init_prog_stack, init_temp_variables, stop = \
            partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack,
                  init_temp_variables, stop, stop_partition, stop_partition_update)

    if not stop:
        init_pointers_pos3 = init_temp_variables[0]
        init_temp_variables = [-1]

    return scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, stop

def quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, stop, stop_partition=False, stop_partition_update=False, stop_quicksort_update=False, stop_save_load_partition=False):

    """ (4 operations)
    POP
    SAVE_LOAD_PARTITION
    PUSH
    STOP

    :param scratchpad_ints:
    :param init_pointers_pos1:
    :param init_pointers_pos2:
    :param init_pointers_pos3:
    :param init_prog_stack:
    :param init_temp_variables:
    :param stop:
    :param stop_partition:
    :param stop_partition_update:
    :param stop_quicksort_update:
    :return:
    """

    if np.random.choice(2, 1, p=[1 - (1 / len(scratchpad_ints)), 1 / len(scratchpad_ints)])[0] == 1 and stop_quicksort_update:
        stop = True

    if not stop:
        init_pointers_pos1 = init_prog_stack.pop()
        init_pointers_pos2 = init_prog_stack.pop()
        init_pointers_pos3 = init_prog_stack.pop()

    if init_pointers_pos1 < init_pointers_pos2 and not stop:

        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, stop = \
            save_load_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3,
                            init_prog_stack, init_temp_variables, stop, stop_partition,
                            stop_partition_update, stop_save_load_partition)

        if not stop:
            init_prog_stack = random_push(init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_temp_variables.copy(), init_prog_stack.copy(), stop)

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy(), stop

def sample_quicksort_indexes(scratchpad_ints, length, sort=False, stop_partition=False, stop_partition_update=False, stop_quicksort_update=False, stop_save_load_partition=False):

    """ (1+n+1)
    PUSH
    from 0 to n:
        QUICKSORT_UPDATE
    STOP

    :param scratchpad_ints:
    :param length:
    :param sort:
    :param stop_partition:
    :param stop_partition_update:
    :param stop_quicksort_update:
    :return:
    """

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
           quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, stop, stop_partition, stop_partition_update, stop_quicksort_update, stop_save_load_partition)

    # This means that we reached the end of the sorting without
    # exiting the loop. Therefore, we initialize everything manually.
    if not stop and not sort:
        np.random.shuffle(scratchpad_ints)
        init_pointers_pos1 = 0
        init_pointers_pos2 = length - 1
        init_pointers_pos3 = 0
        if (stop_quicksort_update):
            init_prog_stack = [0, length-1, 0]
            init_temp_variables = [-1]
        else:
            init_prog_stack = []
            init_temp_variables = [0]

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy()


# Testing
if __name__ == "__main__":
    for i in range(0,10000):
        arr = np.random.randint(0, 100, 5)

        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = \
            sample_quicksort_indexes(np.copy(arr), 5, stop_partition_update=True)
        assert_partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)
        #print(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = \
            sample_quicksort_indexes(np.copy(arr), 5, stop_partition=True)
        assert_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)
        #print(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = \
            sample_quicksort_indexes(np.copy(arr), 5, stop_save_load_partition=True)
        asser_save_load_partition(init_pointers_pos1, init_pointers_pos2)

        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = \
            sample_quicksort_indexes(np.copy(arr), 5, stop_quicksort_update=True)
        assert_quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)
        #print(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = \
            sample_quicksort_indexes(np.copy(arr), 5, sort=True)
        if not np.array_equal(np.array(sorted(arr)), np.array(scratchpad_ints)):
            print("{}, {}".format(np.array(sorted(arr)), np.array(scratchpad_ints)))