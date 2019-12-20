import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from environments.environment import Environment
from environments.utils_quicksort import *


class ListEnvEncoder(nn.Module):
    '''
    Implement an encoder (f_enc) specific to the List environment. It encodes observations e_t into
    vectors s_t of size D = encoding_dim.
    '''

    def __init__(self, observation_dim, encoding_dim):
        super(ListEnvEncoder, self).__init__()
        self.l1 = nn.Linear(observation_dim, 100)
        self.l2 = nn.Linear(100, encoding_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x


class QuickSortListEnv(Environment):
    """Class that represents a list environment. It represents a list of size length of digits. The digits are 10-hot-encoded.
    There are two pointers, each one pointing on a list element. Both pointers can point on the same element.

    The environment state is composed of a scratchpad of size length x 10 which contains the list elements encodings
    and of the two pointers positions.

    An observation is composed of the two encoding of the elements at both pointers positions.

    Primary actions may be called to move pointers and swap elements at their positions.

    We call episode the sequence of actions applied to the environment and their corresponding states.
    The episode stops when the list is sorted.
    """

    def __init__(self, length=10, max_length=10, encoding_dim=32, sample_from_errors_prob=0.3, hierarchy=True, expose_stack=False, validation_mode=False):

        assert length > 0, "length must be a positive integer"
        self.length = length
        self.max_length = max_length
        self.scratchpad_ints = np.zeros((length,))
        self.p1_pos = 0
        self.p2_pos = 0
        self.p3_pos = 0
        self.prog_stack = []
        self.temp_variables = [-1]
        self.encoding_dim = encoding_dim
        self.has_been_reset = False
        self.expose_stack = expose_stack
        self.sample_from_errors_prob = sample_from_errors_prob
        self.max_failed_envs = 100
        self.validation_mode = validation_mode

        self.failed_executions_env = OrderedDict(sorted({
            "PARTITION_UPDATE": [],
            "PARTITION": [],
            "SAVE_LOAD_PARTITION": [],
            "QUICKSORT_UPDATE": [],
            "QUICKSORT": []
                                                        }.items()))

        if hierarchy:
            self.programs_library = OrderedDict(sorted({'STOP': {'level': -1, 'recursive': False},
                                                        'PTR_1_LEFT': {'level': 0, 'recursive': False},
                                                        'PTR_2_LEFT': {'level': 0, 'recursive': False},
                                                        'PTR_3_LEFT': {'level': 0, 'recursive': False},
                                                        'PTR_1_RIGHT': {'level': 0, 'recursive': False},
                                                        'PTR_2_RIGHT': {'level': 0, 'recursive': False},
                                                        'PTR_3_RIGHT': {'level': 0, 'recursive': False},
                                                        'SWAP': {'level': 0, 'recursive': False},
                                                        'SWAP_PIVOT': {'level': 0, 'recursive': False},
                                                        'PUSH': {'level': 0, 'recursive': False},
                                                        'POP': {'level': 0, 'recursive': False},
                                                        'SAVE_PTR_1': {'level': 0, 'recursive': False},
                                                        'LOAD_PTR_1': {'level': 0, 'recursive': False},
                                                        #'RSHIFT': {'level': 1, 'recursive': False},
                                                        #'LSHIFT': {'level': 1, 'recursive': False},
                                                        'PARTITION_UPDATE': {'level': 1, 'recursive': False},
                                                        'PARTITION': {'level': 2, 'recursive': False},
                                                        #'RESET': {'level': 2, 'recursive': False},
                                                        'SAVE_LOAD_PARTITION': {'level': 3, 'recursive': False},
                                                        'QUICKSORT_UPDATE': {'level': 4, 'recursive': False},
                                                        'QUICKSORT': {'level': 5, 'recursive': False}}.items()))
            for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
                self.programs_library[key]['index'] = idx

            self.prog_to_func = OrderedDict(sorted({'STOP': self._stop,
                                                    'PTR_1_LEFT': self._ptr_1_left,
                                                    'PTR_2_LEFT': self._ptr_2_left,
                                                    'PTR_3_LEFT': self._ptr_3_left,
                                                    'PTR_1_RIGHT': self._ptr_1_right,
                                                    'PTR_2_RIGHT': self._ptr_2_right,
                                                    'PTR_3_RIGHT': self._ptr_3_right,
                                                    'SWAP': self._swap,
                                                    'SWAP_PIVOT': self._swap_pivot,
                                                    'PUSH': self._push,
                                                    'POP': self._pop,
                                                    'SAVE_PTR_1': self._save_ptr_1,
                                                    'LOAD_PTR_1': self._load_ptr_1}.items()))

            self.prog_to_precondition = OrderedDict(sorted({'STOP': self._stop_precondition,
                                                            #'RSHIFT': self._rshift_precondition,
                                                            #'LSHIFT': self._lshift_precondition,
                                                            #'RESET': self._reset_precondition,
                                                            'PARTITION_UPDATE': self._partition_update_precondition,
                                                            'PARTITION': self._partition_precondition,
                                                            'SAVE_LOAD_PARTITION': self._save_load_partition_precondition,
                                                            'QUICKSORT_UPDATE': self._quicksort_update_precondition,
                                                            'QUICKSORT': self._quicksort_precondition,
                                                            'PTR_1_LEFT': self._ptr_1_left_precondition,
                                                            'PTR_2_LEFT': self._ptr_2_left_precondition,
                                                            'PTR_3_LEFT': self._ptr_3_left_precondition,
                                                            'PTR_1_RIGHT': self._ptr_1_right_precondition,
                                                            'PTR_2_RIGHT': self._ptr_2_right_precondition,
                                                            'PTR_3_RIGHT': self._ptr_3_right_precondition,
                                                            'SWAP': self._swap_precondition,
                                                            'SWAP_PIVOT': self._swap_pivot_precondition,
                                                            'PUSH': self._push_precondition,
                                                            'POP': self._pop_precondition,
                                                            'SAVE_PTR_1': self._save_ptr_1_precondition,
                                                            'LOAD_PTR_1': self._load_ptr_1_precondition}.items()))

            self.prog_to_postcondition = OrderedDict(sorted({
                                           #'RSHIFT': self._rshift_postcondition,
                                          #'LSHIFT': self._lshift_postcondition,
                                          #'RESET': self._reset_postcondition,
                                          'PARTITION_UPDATE': self._partition_update_postcondition,
                                          'PARTITION': self._partition_postcondition,
                                          'SAVE_LOAD_PARTITION': self._save_load_partition_postcondition,
                                          'QUICKSORT_UPDATE': self._quicksort_update_postcondition,
                                          'QUICKSORT': self._quicksort_postcondition}.items()))

            self.prog_to_structural_condition = OrderedDict(sorted({
                                                                'PARTITION_UPDATE': 'SEQUENTIAL',
                                                                'PARTITION': 'WHILE',
                                                                'SAVE_LOAD_PARTITION': 'SEQUENTIAL',
                                                                'QUICKSORT_UPDATE': 'SEQUENTIAL',
                                                                'QUICKSORT': 'WHILE'}.items()))

        else:
            # In no hierarchy mode, the only non-zero program is Bubblesort

            self.programs_library = {'PTR_1_LEFT': {'level': 0, 'recursive': False},
                                     'STOP': {'level': -1, 'recursive': False},
                                     'PTR_2_LEFT': {'level': 0, 'recursive': False},
                                     'PTR_1_RIGHT': {'level': 0, 'recursive': False},
                                     'PTR_2_RIGHT': {'level': 0, 'recursive': False},
                                     'SWAP': {'level': 0, 'recursive': False},
                                     'BUBBLESORT': {'level': 1, 'recursive': False}}
            for idx, key in enumerate(sorted(list(self.programs_library.keys()))):
                self.programs_library[key]['index'] = idx

            self.prog_to_func = {'STOP': self._stop,
                                 'PTR_1_LEFT': self._ptr_1_left,
                                 'PTR_2_LEFT': self._ptr_2_left,
                                 'PTR_1_RIGHT': self._ptr_1_right,
                                 'PTR_2_RIGHT': self._ptr_2_right,
                                 'SWAP': self._swap}

            self.prog_to_precondition = {'STOP': self._stop_precondition,
                                         'BUBBLESORT': self._bubblesort_precondition,
                                         'PTR_1_LEFT': self._ptr_1_left_precondition,
                                         'PTR_2_LEFT': self._ptr_2_left_precondition,
                                         'PTR_1_RIGHT': self._ptr_1_right_precondition,
                                         'PTR_2_RIGHT': self._ptr_2_right_precondition,
                                         'SWAP': self._swap_precondition}

            self.prog_to_postcondition = {'BUBBLESORT': self._bubblesort_postcondition}

        super(QuickSortListEnv, self).__init__(self.programs_library, self.prog_to_func,
                                               self.prog_to_precondition, self.prog_to_postcondition)

    def _stop(self):
        """Do nothing. The stop action does not modify the environment."""
        pass

    def _stop_precondition(self):
        return True

    def _ptr_1_left(self):
        """Move pointer 1 to the left."""
        if self.p1_pos > 0:
            self.p1_pos -= 1

    def _ptr_1_left_precondition(self):
        return self.p1_pos > 0

    def _ptr_2_left(self):
        """Move pointer 2 to the left."""
        if self.p2_pos > 0:
            self.p2_pos -= 1

    def _ptr_2_left_precondition(self):
        return self.p2_pos > 0

    def _ptr_3_left(self):
        """Move pointer 3 to the left."""
        if self.p3_pos > 0:
            self.p3_pos -= 1

    def _ptr_3_left_precondition(self):
        return self.p3_pos > 0

    def _ptr_1_right(self):
        """Move pointer 1 to the right."""
        if self.p1_pos < (self.length - 1):
            self.p1_pos += 1

    def _ptr_1_right_precondition(self):
        return self.p1_pos < self.length - 1

    def _ptr_2_right(self):
        """Move pointer 2 to the right."""
        if self.p2_pos < (self.length - 1):
            self.p2_pos += 1

    def _ptr_2_right_precondition(self):
        return self.p2_pos < self.length - 1

    def _ptr_3_right(self):
        """Move pointer 3 to the right."""
        if self.p3_pos < (self.length-1):
            self.p3_pos += 1

    def _ptr_3_right_precondition(self):
        return self.p3_pos < self.length - 1

    def _swap(self):
        """Swap the elements pointed by pointers 1 and 2."""
        self.scratchpad_ints[[self.p1_pos, self.p2_pos]] = self.scratchpad_ints[[self.p2_pos, self.p1_pos]]

    def _swap_precondition(self):
        return self.p1_pos != self.p2_pos

    def _swap_pivot(self):
        """Swap the elements pointed by pointers 1 and 3"""
        self.scratchpad_ints[[self.p1_pos, self.p3_pos]] = self.scratchpad_ints[[self.p3_pos, self.p1_pos]]

    def _swap_pivot_precondition(self):
        return self.p1_pos != self.p3_pos #and self.scratchpad_ints[self.p3_pos] < self.scratchpad_ints[self.p2_pos]

    def _push(self):
        if self.p1_pos+1 < self.p2_pos:
            self.prog_stack.append(self.p1_pos+1)
            self.prog_stack.append(self.p2_pos)
            self.prog_stack.append(self.p1_pos+1)
        if self.p1_pos-1 > 0 and self.p3_pos < self.p1_pos - 1:
            self.prog_stack.append(self.p3_pos)
            self.prog_stack.append(self.p1_pos-1)
            self.prog_stack.append(self.p3_pos)

    def _push_precondition(self):
        return self.p1_pos+1 < self.p2_pos or (self.p1_pos-1 > 0 and self.p3_pos < self.p1_pos - 1)

    def _pop(self):
        if len(self.prog_stack) >= 3:
            self.p1_pos = self.prog_stack.pop()
            self.p2_pos = self.prog_stack.pop()
            self.p3_pos = self.prog_stack.pop()

    def _pop_precondition(self):
        return len(self.prog_stack) >=3

    def _save_ptr_1(self):
        self.temp_variables[0] = self.p1_pos

    def _save_ptr_1_precondition(self):
        return True

    def _load_ptr_1(self):
        if self.temp_variables[0] != -1:
            self.p3_pos = self.temp_variables[0]
            self.temp_variables[0] = -1

    def _load_ptr_1_precondition(self):
        return self.temp_variables[0] != -1

    def _compswap_precondition(self):
        bool = self.p1_pos < self.length - 1
        bool &= self.p2_pos == self.p1_pos or self.p2_pos == (self.p1_pos + 1)
        return bool

    def _lshift_precondition(self):
        return self.p1_pos > 0 or self.p2_pos > 0 or self.p3_pos > 0

    def _rshift_precondition(self):
        return self.p1_pos < self.length - 1 or self.p2_pos < self.length - 1 or self.p3_pos < self.length-1

    def _partition_update_precondition(self):
        return self.p3_pos < self.p2_pos and self.p1_pos < self.p2_pos and self.p1_pos <= self.p3_pos and self.temp_variables[0] != -1

    def _partition_precondition(self):
        return self.p1_pos < self.p2_pos and self.p1_pos == self.p3_pos and self.temp_variables[0] == self.p1_pos

    def _save_load_partition_precondition(self):
        return self.p1_pos < self.p2_pos and self.p1_pos == self.p3_pos

    def _quicksort_update_precondition(self):
        return len(self.prog_stack) >= 3 and self.temp_variables[0] == -1

    def _quicksort_precondition(self):
        return len(self.prog_stack) == 0 and self.p1_pos == 0 and self.p2_pos == self.length-1 and self.p3_pos == 0

    def _bubble_precondition(self):
        bool = self.p1_pos == 0
        bool &= ((self.p2_pos == 0) or (self.p2_pos == 1))
        return bool

    def _reset_precondition(self):
        bool = True
        return bool

    def _bubblesort_precondition(self):
        bool = self.p1_pos == 0
        bool &= self.p2_pos == 0
        return bool

    def _compswap_postcondition(self, init_state, state):
        new_scratchpad_ints, new_p1_pos, new_p2_pos = init_state
        new_scratchpad_ints = np.copy(new_scratchpad_ints)
        if new_p1_pos == new_p2_pos and new_p2_pos < self.length - 1:
            new_p2_pos += 1
        idx_left = min(new_p1_pos, new_p2_pos)
        idx_right = max(new_p1_pos, new_p2_pos)
        if new_scratchpad_ints[idx_left] > new_scratchpad_ints[idx_right]:
            new_scratchpad_ints[[idx_left, idx_right]] = new_scratchpad_ints[[idx_right, idx_left]]
        new_state = (new_scratchpad_ints, new_p1_pos, new_p2_pos)
        return self.compare_state(state, new_state)

    def _lshift_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_stack, init_temp_vars = init_state
        scratchpad_ints, p1_pos, p2_pos, p3_pos, stack, temp_vars = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        if init_p1_pos > 0:
            bool &= p1_pos == (init_p1_pos - 1)
        else:
            bool &= p1_pos == init_p1_pos
        if init_p2_pos > 0:
            bool &= p2_pos == (init_p2_pos - 1)
        else:
            bool &= p2_pos == init_p2_pos
        if init_p3_pos > 0:
            bool &= p3_pos == (init_p3_pos - 1)
        else:
            bool &= p3_pos == init_p3_pos
        bool &= (init_stack == stack)
        bool &= (init_temp_vars == temp_vars)
        return bool

    def _rshift_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_stack, init_temp_vars = init_state
        scratchpad_ints, p1_pos, p2_pos, p3_pos, stack, temp_vars = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        if init_p1_pos < self.length - 1:
            bool &= p1_pos == (init_p1_pos + 1)
        else:
            bool &= p1_pos == init_p1_pos
        if init_p2_pos < self.length - 1:
            bool &= p2_pos == (init_p2_pos + 1)
        else:
            bool &= p2_pos == init_p2_pos
        if init_p3_pos < self.length - 1:
            bool &= p3_pos == (init_p3_pos + 1)
        else:
            bool &= p3_pos == init_p3_pos
        bool &= (init_stack == stack)
        bool &= (init_temp_vars == temp_vars)
        return bool

    def _reset_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_stack, init_temp_vars = init_state
        scratchpad_ints, p1_pos, p2_pos, p3_pos, stack, temp_vars = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        bool &= (p1_pos == 0 and p2_pos == 0 and p3_pos == 0)
        bool &= (init_stack == stack)
        bool &= (init_temp_vars == temp_vars)
        return bool

    def _partition_update_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_stack, init_temp_vars = init_state

        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_stack, init_temp_vars \
            = partition_update(init_scratchpad_ints.copy(), init_p1_pos, init_p2_pos, init_p3_pos, init_stack.copy(), init_temp_vars.copy(), sample=False)

        new_state = (np.copy(init_scratchpad_ints), init_p1_pos, init_p2_pos, init_p3_pos, init_stack.copy(), init_temp_vars.copy())
        return self.compare_state(new_state, state)

    def _partition_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_stack, init_temp_vars = init_state

        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_stack, init_temp_vars \
            = partition(init_scratchpad_ints.copy(), init_p1_pos, init_p2_pos, init_p3_pos, init_stack.copy(), init_temp_vars.copy(),
                               sample=False)

        new_state = (np.copy(init_scratchpad_ints), init_p1_pos, init_p2_pos, init_p3_pos, init_stack.copy(), init_temp_vars.copy())
        return self.compare_state(new_state, state)

    def _save_load_partition_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_stack, init_temp_vars = init_state

        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_stack, init_temp_vars \
            = save_load_partition(init_scratchpad_ints.copy(), init_p1_pos, init_p2_pos, init_p3_pos, init_stack.copy(),
                               init_temp_vars.copy(),
                               sample=False)

        new_state = (np.copy(init_scratchpad_ints), init_p1_pos, init_p2_pos, init_p3_pos, init_stack.copy(), init_temp_vars.copy())
        return self.compare_state(new_state, state)

    def _quicksort_update_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_stack, init_temp_vars = init_state

        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_stack, init_temp_vars \
            = quicksort_update(init_scratchpad_ints.copy(), init_p1_pos, init_p2_pos, init_p3_pos, init_stack.copy(), init_temp_vars.copy(),
                               sample=False)

        new_state = (np.copy(init_scratchpad_ints), init_p1_pos, init_p2_pos, init_p3_pos, init_stack.copy(), init_temp_vars.copy())
        return self.compare_state(new_state, state)

    def _quicksort_postcondition(self, init_state, state):
        scratchpad_ints, _, _, _, _, _ = state
        return np.all(scratchpad_ints[:self.length - 1] <= scratchpad_ints[1:self.length])

    def _bubblesort_postcondition(self, init_state, state):
        scratchpad_ints, p1_pos, p2_pos = state
        # check if list is sorted
        return np.all(scratchpad_ints[:self.length - 1] <= scratchpad_ints[1:self.length])

    def _bubble_postcondition(self, init_state, state):
        new_scratchpad_ints, new_p1_pos, new_p2_pos = init_state
        new_scratchpad_ints = np.copy(new_scratchpad_ints)
        for idx in range(0, self.length - 1):
            if new_scratchpad_ints[idx + 1] < new_scratchpad_ints[idx]:
                new_scratchpad_ints[[idx, idx + 1]] = new_scratchpad_ints[[idx + 1, idx]]
        # bubble is expected to terminate with both pointers at the extreme left of the list
        new_p1_pos = self.length - 1
        new_p2_pos = self.length - 1
        new_state = (new_scratchpad_ints, new_p1_pos, new_p2_pos)
        return self.compare_state(state, new_state)

    def _one_hot_encode(self, digit, basis=10):
        """One hot encode a digit with basis.

        Args:
          digit: a digit (integer between 0 and 9)
          basis:  (Default value = 10)

        Returns:
          a numpy array representing the 10-hot-encoding of the digit

        """
        encoding = np.zeros(basis)
        encoding[digit] = 1
        return encoding

    def _one_hot_decode(self, one_encoding):
        """Returns digit associated to a one hot encoding.

        Args:
          one_encoding: numpy array representing the 10-hot-encoding of a digit.

        Returns:
          the digit encoded in one_encoding

        """
        return np.argmax(one_encoding)

    def update_failing_envs(self, state, program):
        """
        Update failing env count
        :param env: current failed env
        :param program: current failed program
        :return:
        """

        # Do not update if we are running in validation mode
        if self.validation_mode:
            return

        if len(self.failed_executions_env[program]) == 0:
            self.failed_executions_env[program].append((self.get_state_clone(state), 1, 1000))
        else:
            found = False
            for i in range(len(self.failed_executions_env[program])):
                if self.compare_state(state, self.failed_executions_env[program][i][0]):
                    self.failed_executions_env[program][i] = (self.failed_executions_env[program][i][0], self.failed_executions_env[program][i][1]+1, self.failed_executions_env[program][i][2]+1)
                    found = True
                    break
                else:
                    self.failed_executions_env[program][i] = (self.failed_executions_env[program][i][0], self.failed_executions_env[program][i][1], self.failed_executions_env[program][i][2]-1)
            if not found:
                # Remove the failed program with the least life from the list to make space for the new one
                if len(self.failed_executions_env[program]) >= self.max_failed_envs:
                    self.failed_executions_env[program].sort(key=lambda t: t[2])
                    print(self.failed_executions_env[program][0])
                    del self.failed_executions_env[program][0]
                self.failed_executions_env[program].append((self.get_state_clone(state), 1, 1000))

    def return_sample_state(self, program):
        """
        Return the dictionary where to sample from.
        :param program: program we are resetting
        :return: the dictionary
        """
        if np.random.random_sample() < self.sample_from_errors_prob \
                and len(self.failed_executions_env[program]) > 0 \
                and not self.validation_mode:
            env = self.failed_executions_env
            total_errors = sum([x[1] for x in env[program]])
            sampling_prob = [x[1]/total_errors for x in env[program]]
            index = np.random.choice(len(env[program]), p=sampling_prob)
            result = env[program][index][0]

            # This accounts for the fact that we are maybe sampling from a state
            # which is smaller than our current scratchpad length.
            self.length = len(result[0])
        else:
            env = self.sampled_env
            index = np.random.choice(len(env[program]))
            result = env[program][index]

        return len(env[program]), result

    def reset_env(self):
        """Reset the environment. The list are values are draw randomly. The pointers are initialized at position 0
        (at left position of the list).

        """
        current_task_name = self.get_program_from_index(self.current_task_index)

        total_size=-1
        index = -1

        # Sample multiple time the env if some of the steps is empty.
        redo = True
        while redo:
            redo = False
            self.scratchpad_ints = np.random.randint(10, size=self.length)
            self.sampled_env = sample_quicksort_indexes(np.copy(self.scratchpad_ints), self.length)
            for v in self.sampled_env.values():
                if len(v) == 0:
                    redo = True

        # Randomly initialize things TODO: this can be probably removed.
        if (int(np.random.randint(0,10)%2==0)):
            init_prog_stack = list(np.random.random_integers(0, self.length-1, 3))
        else:
            init_prog_stack = []
        init_temp_variables = [-1 if np.random.randint(0, 2) == 1 else 0]

        if current_task_name == 'BUBBLE' or current_task_name == 'BUBBLESORT':
            init_pointers_pos1 = 0
            init_pointers_pos2 = 0
            init_pointers_pos3 = 0
        elif current_task_name == 'PARTITION':

            total_size, env = self.return_sample_state("PARTITION")
            temp_scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, \
            init_prog_stack, init_temp_variables = env
            self.scratchpad_ints = np.copy(temp_scratchpad_ints)

        elif current_task_name == 'PARTITION_UPDATE':

            total_size, env = self.return_sample_state("PARTITION_UPDATE")
            temp_scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, \
            init_prog_stack, init_temp_variables = env
            self.scratchpad_ints = np.copy(temp_scratchpad_ints)

        elif current_task_name == 'SAVE_LOAD_PARTITION':

            total_size, env = self.return_sample_state("SAVE_LOAD_PARTITION")
            temp_scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, \
            init_prog_stack, init_temp_variables = env
            self.scratchpad_ints = np.copy(temp_scratchpad_ints)

        elif current_task_name == 'QUICKSORT_UPDATE':

            total_size, env = self.return_sample_state("QUICKSORT_UPDATE")
            temp_scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, \
            init_prog_stack, init_temp_variables = env
            self.scratchpad_ints = np.copy(temp_scratchpad_ints)

        elif current_task_name == 'QUICKSORT':

            # Here the first value is the initial condition
            index = 0
            temp_scratchpad_ints, init_pointers_pos1, init_pointers_pos2, init_pointers_pos3, \
            init_prog_stack, init_temp_variables = self.sampled_env["QUICKSORT"][0]
            self.scratchpad_ints = np.copy(temp_scratchpad_ints)

        elif current_task_name == 'RESET':
            while True:
                init_pointers_pos1 = int(np.random.randint(0, self.length))
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                init_pointers_pos3 = int(np.random.randint(0, self.length))
                if not (init_pointers_pos1 == 0 and init_pointers_pos2 == 0 and init_pointers_pos3 == 0):
                    break
        elif current_task_name == 'LSHIFT':
            while True:
                init_pointers_pos1 = int(np.random.randint(0, self.length))
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                init_pointers_pos3 = int(np.random.randint(0, self.length))
                if not (init_pointers_pos1 == 0 and init_pointers_pos2 == 0 and init_pointers_pos3 == 0):
                    break
        elif current_task_name == 'RSHIFT':
            while True:
                init_pointers_pos1 = int(np.random.randint(0, self.length))
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                init_pointers_pos3 = int(np.random.randint(0, self.length))
                if not (init_pointers_pos1 == self.length - 1 and init_pointers_pos2 == self.length - 1 and init_pointers_pos3 == self.length-1):
                    break
        elif current_task_name == 'COMPSWAP':
            init_pointers_pos1 = int(np.random.randint(0, self.length - 1))
            init_pointers_pos2 = int(np.random.choice([init_pointers_pos1, init_pointers_pos1 + 1]))
            init_pointers_pos3 = 0
        else:
            raise NotImplementedError('Unable to reset env for this program...')

        self.p1_pos = init_pointers_pos1
        self.p2_pos = init_pointers_pos2
        self.p3_pos = init_pointers_pos3
        self.prog_stack = init_prog_stack.copy()
        self.temp_variables = init_temp_variables.copy()
        self.has_been_reset = True

        return index, total_size

    def get_state(self):
        """Returns the current state.

        Returns:
            the environment state

        """
        assert self.has_been_reset, 'Need to reset the environment before getting states'
        return np.copy(self.scratchpad_ints), self.p1_pos, self.p2_pos, self.p3_pos, self.prog_stack.copy(), self.temp_variables.copy()

    def get_observation(self):
        """Returns an observation of the current state.

        Returns:
            an observation of the current state
        """
        assert self.has_been_reset, 'Need to reset the environment before getting observations'

        p1_val = self.scratchpad_ints[self.p1_pos]
        p2_val = self.scratchpad_ints[self.p2_pos]
        p3_val = self.scratchpad_ints[self.p3_pos]
        is_sorted = int(self._is_sorted())
        is_stack_full = int(len(self.prog_stack) >= 3)
        is_ptr1_saved = int(self.temp_variables[0] != -1)
        pointers_same_pos = int(self.p1_pos == self.p2_pos)
        pointers_same_pos_2 = int(self.p2_pos == self.p3_pos)
        pointers_same_pos_3 = int(self.p3_pos == self.p1_pos)
        is_pointer_1_less_than_2 = int(self.p1_pos < self.p2_pos)
        is_pointer_3_less_than_2 = int(self.p3_pos < self.p2_pos)
        pt_1_left = int(self.p1_pos == 0)
        pt_2_left = int(self.p2_pos == 0)
        pt_3_left = int(self.p3_pos == 0)
        pt_1_right = int(self.p1_pos == (self.length - 1))
        pt_2_right = int(self.p2_pos == (self.length - 1))
        pt_3_right = int(self.p3_pos == (self.length - 1))
        p1p2p3 = np.eye(10)[[p1_val, p2_val, p3_val]].reshape(-1)
        #p1p2p3_pos = np.array([self.p1_pos, self.p2_pos, self.p3_pos])
        #first_stack_pos = np.array([self.prog_stack[len(self.prog_stack)-2], self.prog_stack[len(self.prog_stack)-1]]) if is_stack_full else np.array([-1, -1])
        #how_many_pointers_saved = np.array([len(self.prog_stack)/3])

        if self.expose_stack:
            if is_stack_full:
                first_stack_elem = self.prog_stack[len(self.prog_stack)-2]
                second_stack_elem = self.prog_stack[len(self.prog_stack)-1]
                topstack = np.eye(10)[[self.scratchpad_ints[first_stack_elem], self.scratchpad_ints[second_stack_elem]]].reshape(-1)
            else:
                topstack = np.eye(10)[[0,0]].reshape(-1)

        bools = np.array([
            pt_1_left,
            pt_1_right,
            pt_2_left,
            pt_2_right,
            pt_3_left,
            pt_3_right,
            pointers_same_pos,
            pointers_same_pos_2,
            pointers_same_pos_3,
            is_sorted,
            is_stack_full,
            is_ptr1_saved,
            is_pointer_1_less_than_2,
            is_pointer_3_less_than_2
        ])
        #return np.concatenate((p1p2p3, p1p2p3_pos, first_stack_pos, bools), axis=0)

        # If we want to expose the stack then we concatenate it
        if self.expose_stack:
            final_observation = np.concatenate((p1p2p3, topstack, bools), axis=0)
        else:
            final_observation = np.concatenate((p1p2p3, bools), axis=0)

        return final_observation

    def get_observation_dim(self):
        """

        Returns:
            the size of the observation tensor
        """
        #return 3 * 10 + 3 + 2 + 12
        if self.expose_stack:
            total_observation_dim = 3*10 + 2*10 + 14
        else:
            total_observation_dim = 3 * 10 + 14

        return total_observation_dim

    def reset_to_state(self, state):
        """

        Args:
          state: a given state of the environment
        reset the environment is the given state

        """
        self.scratchpad_ints = state[0].copy()
        self.p1_pos = state[1]
        self.p2_pos = state[2]
        self.p3_pos = state[3]
        self.prog_stack = state[4].copy()
        self.temp_variables = state[5].copy()

    def _is_sorted(self):
        """Assert is the list is sorted or not.

        Args:

        Returns:
            True if the list is sorted, False otherwise

        """
        arr = self.scratchpad_ints
        return np.all(arr[:-1] <= arr[1:])

    def get_state_str(self, state):
        """Print a graphical representation of the environment state"""
        scratchpad = state[0].copy()  # check
        p1_pos = state[1]
        p2_pos = state[2]
        p3_pos = state[3]
        stack = state[4].copy()
        temp = state[5].copy()
        str = 'list: {}, p1 : {}, p2 : {}, p3 : {}, stack: {}, temp_vars: {}'.format(scratchpad, p1_pos, p2_pos, p3_pos, stack, temp)
        return str

    def compare_state(self, state1, state2):
        """
        Compares two states.

        Args:
            state1: a state
            state2: a state

        Returns:
            True if both states are equals, False otherwise.

        """
        bool = True
        bool &= np.array_equal(state1[0], state2[0])
        bool &= (state1[1] == state2[1])
        bool &= (state1[2] == state2[2])
        bool &= (state1[3] == state2[3])
        bool &= (state1[4] == state2[4])
        bool &= (state1[5] == state2[5])
        return bool

    def get_state_clone(self, state):
        """
        Get a clone of the current state
        :return:
        """
        return np.copy(state[0]), state[1], state[2], state[3], state[4].copy(), state[5].copy()