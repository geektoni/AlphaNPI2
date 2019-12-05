import numpy as np

from collections import OrderedDict

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

def assert_save_load_partition(init_pointers_pos1, init_pointers_pos2):
    assert init_pointers_pos1 < init_pointers_pos2 and init_pointers_pos1 == init_pointers_pos3, "Save Load Partition {}, {}".format(init_pointers_pos1, init_pointers_pos2)

def random_push(init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_temp_variables, init_prog_stack, stop, randomize=False):

    val = np.random.randint(0,2) if randomize else 1

    if val == 0:
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
    assert len(stack) >= 3 and temp[0] == -1, "Quicksort Update: {}".format(stack)

def partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, sampled_environment={}, sample=True):

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

    if sample:
        sampled_environment["PARTITION_UPDATE"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack.copy(), temp.copy()))

    if scratchpad_ints[init_pointers_pos3] < scratchpad_ints[init_pointers_pos2]:
        scratchpad_ints[[init_pointers_pos3, init_pointers_pos1]] = scratchpad_ints[
            [init_pointers_pos1, init_pointers_pos3]]
        init_pointers_pos1 += 1

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack.copy(), temp.copy()


def partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, sampled_environment={}, sample=True):
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

    if sample:
        sampled_environment["PARTITION"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack.copy(), temp.copy()))

    while init_pointers_pos3 < init_pointers_pos2:
        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = \
        partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp, sampled_environment=sampled_environment, sample=sample)
        init_pointers_pos3 += 1

    scratchpad_ints[[init_pointers_pos1, init_pointers_pos2]] = scratchpad_ints[[init_pointers_pos2, init_pointers_pos1]]

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack.copy(), temp.copy()

def save_load_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, sampled_environment={}, sample=True):

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

    if sample:
        sampled_environment["SAVE_LOAD_PARTITION"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy()))

    init_temp_variables = [init_pointers_pos1]

    # Run the partition method
    scratchpad_ints, init_pointers_pos1, init_pointers_pos2, \
    init_pointers_pos3, init_prog_stack, init_temp_variables = \
        partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack,
                init_temp_variables, sampled_environment, sample=sample)

    init_pointers_pos3 = init_temp_variables[0]
    init_temp_variables = [-1]

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy()

def quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, randomize=False, sampled_environment={}, sample=True):

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

    if sample:
        sampled_environment["QUICKSORT_UPDATE"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy()))

    init_pointers_pos1 = init_prog_stack.pop()
    init_pointers_pos2 = init_prog_stack.pop()
    init_pointers_pos3 = init_prog_stack.pop()

    if init_pointers_pos1 < init_pointers_pos2:

        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables = \
            save_load_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3,
                            init_prog_stack, init_temp_variables, sampled_environment, sample=sample)

        init_prog_stack = random_push(init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_temp_variables.copy(), init_prog_stack.copy(), randomize)

    return np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy()

def sample_quicksort_indexes(scratchpad_ints, length, randomize_push=False):

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

    sampled_environment = OrderedDict(sorted({"QUICKSORT": [],
                                              "PARTITION_UPDATE": [],
                                              "PARTITION": [],
                                              "SAVE_LOAD_PARTITION": [],
                                              "QUICKSORT_UPDATE": []}.items()))

    init_pointers_pos1 = 0
    init_pointers_pos2 = length - 1
    init_pointers_pos3 = 0

    init_temp_variables = [-1]

    init_prog_stack = []
    init_prog_stack.append(init_pointers_pos3)
    init_prog_stack.append(init_pointers_pos2)
    init_prog_stack.append(init_pointers_pos1)

    sampled_environment["QUICKSORT"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy()))

    while len(init_prog_stack) > 0:
        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables = \
           quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack, init_temp_variables, randomize_push, sampled_environment)

    sampled_environment["QUICKSORT"].append((np.copy(scratchpad_ints), init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, init_prog_stack.copy(), init_temp_variables.copy()))

    return sampled_environment.copy()


# Testing
if __name__ == "__main__":
    for i in range(0,10000):

        arr = np.random.randint(0, 100, 10)

        env = sample_quicksort_indexes(np.copy(arr), 10)


        for e in env["PARTITION_UPDATE"]:
            scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = e
            assert_partition_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

        for e in env["PARTITION"]:
            scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = e
            assert_partition(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

        for e in env["SAVE_LOAD_PARTITION"]:
            scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = e
            assert_save_load_partition(init_pointers_pos1, init_pointers_pos2)

        for e in env["QUICKSORT_UPDATE"]:
            scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = e
            assert_quicksort_update(scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp)

        scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, stack, temp = env["QUICKSORT"][1]

        if not np.all(scratchpad_ints[:len(scratchpad_ints) - 1] <= scratchpad_ints[1:len(scratchpad_ints)]):
            print("Not Sorted")

        if not np.array_equal(np.array(sorted(arr)), np.array(scratchpad_ints)):
            print("{}, {}".format(np.array(sorted(arr)), np.array(scratchpad_ints)))