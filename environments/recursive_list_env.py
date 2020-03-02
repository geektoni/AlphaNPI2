import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from environments.environment import Environment


class RecursiveListEnvEncoder(nn.Module):
    '''
    Implement an encoder (f_enc) specific to the List environment. It encodes observations e_t into
    vectors s_t of size D = encoding_dim.
    '''

    def __init__(self, observation_dim, encoding_dim):
        super(RecursiveListEnvEncoder, self).__init__()
        self.l1 = nn.Linear(observation_dim, 100)
        self.l2 = nn.Linear(100, encoding_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x


class RecursiveListEnv(Environment):
    def __init__(self, length=10, encoding_dim=32):

        assert length > 1, "length must be a positive integer superior to 1"
        self.length = length
        self.start_pos = 0
        self.end_pos = length-1
        self.scratchpad_ints = np.zeros((length,))
        self.p1_pos = 0
        self.p2_pos = 0
        self.encoding_dim = encoding_dim
        self.has_been_reset = False

        self.programs_library = {'PTR_1_LEFT': {'level': 0, 'recursive': False},
                                 'STOP': {'level': -1, 'recursive': False},
                                 'PTR_2_LEFT': {'level': 0, 'recursive': False},
                                 'PTR_1_RIGHT': {'level': 0, 'recursive': False},
                                 'PTR_2_RIGHT': {'level': 0, 'recursive': False},
                                 'SWAP': {'level': 0, 'recursive': False},
                                 'RSHIFT': {'level': 1, 'recursive': False},
                                 'LSHIFT': {'level': 1, 'recursive': False},
                                 'COMPSWAP': {'level': 1, 'recursive': False},
                                 'RESET': {'level': 2, 'recursive': True},
                                 'BUBBLE': {'level': 2, 'recursive': True},
                                 'BUBBLESORT': {'level': 3, 'recursive': True}}

        for i, key in enumerate(sorted(list(self.programs_library.keys()))):
            self.programs_library[key]['index'] = i

        self.prog_to_func = {'STOP': self._stop,
                             'PTR_1_LEFT': self._ptr_1_left,
                             'PTR_2_LEFT': self._ptr_2_left,
                             'PTR_1_RIGHT': self._ptr_1_right,
                             'PTR_2_RIGHT': self._ptr_2_right,
                             'SWAP': self._swap}

        self.prog_to_precondition = {'STOP': self._stop_precondition,
                                     'RSHIFT': self._rshift_precondition,
                                     'LSHIFT': self._lshift_precondition,
                                     'COMPSWAP': self._compswap_precondition,
                                     'RESET': self._reset_precondition,
                                     'BUBBLE': self._bubble_precondition,
                                     'BUBBLESORT': self._bubblesort_precondition,
                                     'PTR_1_LEFT': self._ptr_1_left_precondition,
                                     'PTR_2_LEFT': self._ptr_2_left_precondition,
                                     'PTR_1_RIGHT': self._ptr_1_right_precondition,
                                     'PTR_2_RIGHT': self._ptr_2_right_precondition,
                                     'SWAP': self._swap_precondition}

        self.prog_to_postcondition = {'RSHIFT': self._rshift_postcondition,
                                     'LSHIFT': self._lshift_postcondition,
                                     'COMPSWAP': self._compswap_postcondition,
                                     'RESET': self._reset_postcondition,
                                     'BUBBLE': self._bubble_postcondition,
                                     'BUBBLESORT': self._bubblesort_postcondition}

        super(RecursiveListEnv, self).__init__(self.programs_library, self.prog_to_func,
                                               self.prog_to_precondition, self.prog_to_postcondition)

    def _decr_length_right(self):
        assert self._decr_length_right_precondition(), 'precondition not verified'
        if self.end_pos > self.start_pos:
            self.end_pos -= 1

    def _decr_length_right_precondition(self):
        return self.end_pos > self.start_pos and (self.p1_pos < self.end_pos and self.p2_pos < self.end_pos)

    def _decr_length_left(self):
        assert self._decr_length_left_precondition(), 'precondition not verified'
        if self.start_pos < self.end_pos:
            self.start_pos += 1

    def _decr_length_left_precondition(self):
        return self.start_pos < self.end_pos and (self.p1_pos > self.start_pos and self.p2_pos > self.start_pos)

    def _incr_length_left(self):
        assert self._incr_length_left_precondition(), 'precondition not verified'
        if self.start_pos > 0:
            self.start_pos -= 1

    def _incr_length_left_precondition(self):
        return self.start_pos > 0

    def _incr_length_right(self):
        assert self._incr_length_right_precondition(), 'precondition not verified'
        if self.end_pos < self.length-1:
            self.end_pos += 1

    def _incr_length_right_precondition(self):
        return self.end_pos < self.length-1

    def _ptr_1_left(self):
        assert self._ptr_1_left_precondition(), 'precondition not verified'
        if self.p1_pos > self.start_pos:
            self.p1_pos -= 1

    def _ptr_1_left_precondition(self):
        return self.p1_pos > self.start_pos

    def _stop(self):
        assert self._stop_precondition(), 'precondition not verified'
        pass

    def _stop_precondition(self):
        return True

    def _ptr_2_left(self):
        assert self._ptr_2_left_precondition(), 'precondition not verified'
        if self.p2_pos > self.start_pos:
            self.p2_pos -= 1

    def _ptr_2_left_precondition(self):
        return self.p2_pos > self.start_pos

    def _ptr_1_right(self):
        assert self._ptr_1_right_precondition(), 'precondition not verified'
        if self.p1_pos < self.end_pos:
            self.p1_pos += 1

    def _ptr_1_right_precondition(self):
        return self.p1_pos < self.end_pos

    def _ptr_2_right(self):
        assert self._ptr_2_right_precondition(), 'precondition not verified'
        if self.p2_pos < self.end_pos:
            self.p2_pos += 1

    def _ptr_2_right_precondition(self):
        return self.p2_pos < self.end_pos

    def _swap(self):
        assert self._swap_precondition(), 'precondition not verified'
        self.scratchpad_ints[[self.p1_pos, self.p2_pos]] = self.scratchpad_ints[[self.p2_pos, self.p1_pos]]

    def _swap_precondition(self):
        return self.p1_pos != self.p2_pos

    def _compswap_precondition(self):
        list_length = self.end_pos - self.start_pos + 1
        bool = list_length > 1
        bool &= self.p1_pos < self.end_pos
        bool &= self.p2_pos == self.p1_pos or self.p2_pos == (self.p1_pos + 1)
        return bool

    def _compswap_postcondition(self, init_state, state):
        new_scratchpad_ints, new_p1_pos, new_p2_pos, new_start_pos, new_end_pos = init_state
        new_scratchpad_ints = np.copy(new_scratchpad_ints)
        if new_p1_pos == new_p2_pos and new_p2_pos < new_end_pos:
            new_p2_pos += 1
        idx_left = min(new_p1_pos, new_p2_pos)
        idx_right = max(new_p1_pos, new_p2_pos)
        if new_scratchpad_ints[idx_left] > new_scratchpad_ints[idx_right]:
            new_scratchpad_ints[[idx_left, idx_right]] = new_scratchpad_ints[[idx_right, idx_left]]
        new_state = (new_scratchpad_ints, new_p1_pos, new_p2_pos, new_start_pos, new_end_pos)
        return self.compare_state(state, new_state)

    def _lshift_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_start_pos, init_end_pos = init_state
        scratchpad_ints, p1_pos, p2_pos, start_pos, end_pos = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        bool &= init_start_pos == start_pos
        bool &= init_end_pos == end_pos
        if init_p1_pos > init_start_pos:
            bool &= p1_pos == (init_p1_pos-1)
        else:
            bool &= p1_pos == init_p1_pos

        if init_p2_pos > init_start_pos:
            bool &= p2_pos == (init_p2_pos-1)
        else:
            bool &= p2_pos == init_p2_pos

        return bool

    def _rshift_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_start_pos, init_end_pos = init_state
        scratchpad_ints, p1_pos, p2_pos, start_pos, end_pos = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        bool &= init_start_pos == start_pos
        bool &= init_end_pos == end_pos
        if init_p1_pos < init_end_pos:
            bool &= p1_pos == (init_p1_pos+1)
        else:
            bool &= p1_pos == init_p1_pos
        if init_p2_pos < init_end_pos:
            bool &= p2_pos == (init_p2_pos+1)
        else:
            bool &= p2_pos == init_p2_pos
        return bool

    def _reset_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_start_pos, init_end_pos = init_state
        scratchpad_ints, p1_pos, p2_pos, start_pos, end_pos = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        bool &= init_start_pos == start_pos
        bool &= init_end_pos == end_pos
        bool &= (p1_pos == start_pos and p2_pos == start_pos)
        return bool

    def _bubblesort_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_start_pos, init_end_pos = init_state
        scratchpad_ints, p1_pos, p2_pos, start_pos, end_pos = state
        bool = init_start_pos == start_pos
        bool &= init_end_pos == end_pos
        # check if list is sorted
        bool &= np.all(scratchpad_ints[:end_pos] <= scratchpad_ints[(start_pos+1):(end_pos+1)])
        return bool

    def _bubble_postcondition(self, init_state, state):
        new_scratchpad_ints, new_p1_pos, new_p2_pos, new_start_pos, new_end_pos = init_state
        new_scratchpad_ints = np.copy(new_scratchpad_ints)
        for idx in range(new_start_pos, new_end_pos):
            if new_scratchpad_ints[idx+1] < new_scratchpad_ints[idx]:
                new_scratchpad_ints[[idx, idx+1]] = new_scratchpad_ints[[idx+1, idx]]
        new_p1_pos = new_end_pos
        new_p2_pos = new_end_pos
        new_state = (new_scratchpad_ints, new_p1_pos, new_p2_pos, new_start_pos, new_end_pos)
        return self.compare_state(state, new_state)

    def _lshift_precondition(self):
        return self.p1_pos > self.start_pos or self.p2_pos > self.start_pos

    def _rshift_precondition(self):
        return self.p1_pos < self.end_pos or self.p2_pos < self.end_pos

    def _bubble_precondition(self):
        bubble_index = self.programs_library['BUBBLE']['index']
        if self.current_task_index != bubble_index:
            bool = self.p1_pos == self.start_pos
            bool &= ((self.p2_pos == self.start_pos) or (self.p2_pos == (self.start_pos+1)))
        else:
            bool = self.p1_pos == (self.start_pos + 1)
            bool &= ((self.p2_pos == self.start_pos) or (self.p2_pos == (self.start_pos + 1)))
            bool &= self._decr_length_left_precondition()
        return bool

    def _reset_precondition(self):
        bool = (self.p1_pos > self.start_pos or self.p2_pos > self.start_pos)
        reset_index = self.programs_library['RESET']['index']
        if self.current_task_index == reset_index:
            bool &= self._decr_length_right_precondition()
        return bool

    def _bubblesort_precondition(self):
        bool = self.p1_pos == self.start_pos
        bool &= self.p2_pos == self.start_pos
        bubblesort_index = self.programs_library['BUBBLESORT']['index']
        if self.current_task_index == bubblesort_index:
            bool &= self._decr_length_right_precondition()
        return bool

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
          a: numpy array representing the 10-hot-encoding of a digit.
          one_encoding:

        Returns:
          the digit encoded in one_encoding

        """
        return np.argmax(one_encoding)

    def reset_env(self):
        """Reset the environment. The list are values are draw randomly. The pointers are initialized at position 0
        (at left position of the list).

        """

        total_size = -1
        index = -1

        assert self.length > 1, "list length must be greater than 1"
        self.start_pos = 0
        self.end_pos = self.length-1
        self.scratchpad_ints = np.random.randint(10, size=self.length)
        current_task_name = self.get_program_from_index(self.current_task_index)
        if current_task_name == 'BUBBLE' or current_task_name == 'BUBBLESORT':
            init_pointers_pos1 = 0
            init_pointers_pos2 = 0
        elif current_task_name == 'RESET':
            while True:
                init_pointers_pos1 = int(np.random.randint(0, self.length))
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                if not (init_pointers_pos1 == 0 and init_pointers_pos2 == 0):
                    break
        elif current_task_name == 'LSHIFT':
            while True:
                init_pointers_pos1 = int(np.random.randint(0, self.length))
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                if not (init_pointers_pos1 == 0 and init_pointers_pos2 == 0):
                    break
        elif current_task_name == 'RSHIFT':
            while True:
                init_pointers_pos1 = int(np.random.randint(0, self.length))
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                if not (init_pointers_pos1 == self.length - 1 and init_pointers_pos2 == self.length - 1):
                    break
        elif current_task_name == 'COMPSWAP':
            init_pointers_pos1 = int(np.random.randint(0, self.length - 1))
            init_pointers_pos2 = int(np.random.choice([init_pointers_pos1, init_pointers_pos1 + 1]))
        else:
            raise NotImplementedError('Unable to reset env for this program...')

        self.p1_pos = init_pointers_pos1
        self.p2_pos = init_pointers_pos2
        self.has_been_reset = True

        return index, total_size

    def get_observation(self):
        """Returns an observation of the current state.

        Returns:
            an observation of the current state
        """
        assert self.has_been_reset, 'Need to reset the environment before getting observations'

        p1_val = self.scratchpad_ints[self.p1_pos]
        p2_val = self.scratchpad_ints[self.p2_pos]
        is_sorted = int(self._is_sorted())
        pointers_same_pos = int(self.p1_pos == self.p2_pos)
        pt_1_left = int(self.p1_pos == self.start_pos)
        pt_2_left = int(self.p2_pos == self.start_pos)
        pt_1_right = int(self.p1_pos == self.end_pos)
        pt_2_right = int(self.p2_pos == self.end_pos)
        p1p2 = np.eye(10)[[p1_val, p2_val]].reshape(-1)  # one hot encoding of values at pointers pos
        bools = np.array([
            pt_1_left,
            pt_1_right,
            pt_2_left,
            pt_2_right,
            pointers_same_pos,
            is_sorted
        ])
        return np.concatenate((p1p2, bools), axis=0)

    def get_observation_dim(self):
        """

        Returns:
            the size of the observation tensor
        """
        return 2 * 10 + 6

    def get_state(self):
        """Returns the current state.

        Returns:
            the environment state

        """
        assert self.has_been_reset, 'Need to reset the environment before getting states'
        return np.copy(self.scratchpad_ints), self.p1_pos, self.p2_pos, self.start_pos, self.end_pos

    def reset_to_state(self, state):
        """

        Args:
          state: a given state of the environment
        reset the environment is the given state

        """
        self.scratchpad_ints = state[0].copy()
        self.p1_pos = state[1]
        self.p2_pos = state[2]
        self.start_pos = state[3]
        self.end_pos = state[4]

    def get_state_str(self, state):
        """Print a graphical representation of the environment state"""
        scratchpad = state[0].copy()
        p1_pos = state[1]
        p2_pos = state[2]
        start_pos = state[3]
        end_pos = state[4]
        scratchpad = scratchpad[start_pos:end_pos+1]
        str = 'list: {}, p1 : {}, p2 : {}, start_pos: {}, end_pos: {}'.format(scratchpad, p1_pos,
                                                                              p2_pos, start_pos, end_pos)
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
        bool = np.array_equal(state1[0], state2[0])
        bool &= (state1[1] == state2[1])
        bool &= (state1[2] == state2[2])
        bool &= (state1[3] == state2[3])
        bool &= (state1[4] == state2[4])
        return bool

    def _is_sorted(self):
        """Assert is the list is sorted or not.

        Args:

        Returns:
            True if the list is sorted, False otherwise

        """
        arr = self.scratchpad_ints
        return np.all(arr[:self.length-1] <= arr[1:self.length])

    def start_task(self, task_index):
        if self.tasks_list.count(task_index) > 0:
            task = self.get_program_from_index(task_index)
            if task == 'RESET' or task == 'BUBBLESORT':
                self._decr_length_right()
            if task == 'BUBBLE':
                self._decr_length_left()
        obs, sindex, ssize = super(RecursiveListEnv, self).start_task(task_index)
        return obs, sindex, ssize

    def end_task(self):
        current_task = self.get_program_from_index(self.current_task_index)
        if current_task == 'RESET' or current_task == 'BUBBLESORT':
            if self.tasks_list.count(self.current_task_index) > 1:
                self._incr_length_right()
        if current_task == 'BUBBLE':
            if self.tasks_list.count(self.current_task_index) > 1:
                self._incr_length_left()
        super(RecursiveListEnv, self).end_task()

    def update_failing_envs(self, state, program_name):
        pass

class QuickSortRecursiveListEnv(Environment):
    def __init__(self, length=10, encoding_dim=32):

        assert length > 1, "length must be a positive integer superior to 1"
        self.length = length
        self.start_pos = 0
        self.end_pos = length-1
        self.scratchpad_ints = np.zeros((length,))
        self.p1_pos = 0
        self.p2_pos = 0
        self.p3_pos = 0
        self.prog_stack = []
        self.encoding_dim = encoding_dim
        self.has_been_reset = False

        self.programs_library = OrderedDict(sorted({
                                 'STOP': OrderedDict({'level': -1, 'recursive': False}),
                                 'PTR_1_LEFT': OrderedDict({'level': 0, 'recursive': False}),
                                 'PTR_2_LEFT': OrderedDict({'level': 0, 'recursive': False}),
                                 'PTR_1_RIGHT': OrderedDict({'level': 0, 'recursive': False}),
                                 'PTR_2_RIGHT': OrderedDict({'level': 0, 'recursive': False}),
                                 'PTR_3_LEFT': OrderedDict({'level': 0, 'recursive': False}),
                                 'PTR_3_RIGHT': OrderedDict({'level': 0, 'recursive': False}),
                                 'SWAP': OrderedDict({'level': 0, 'recursive': False}),
                                 'SWAP_2': OrderedDict({'level': 0, 'recursive': False}),
                                 'SWAP_3': OrderedDict({'level': 0, 'recursive': False}),
                                 'PUSH': OrderedDict({'level': 0, 'recursive': False}),
                                 'POP': OrderedDict({'level': 0, 'recursive': False}),
                                 'RSHIFT': OrderedDict({'level': 1, 'recursive': False}),
                                 'LSHIFT': OrderedDict({'level': 1, 'recursive': False}),
                                 'PARTITION_UPDATE': OrderedDict({'level': 1, 'recursive': False}),
                                 'RESET': OrderedDict({'level': 2, 'recursive': True}),
                                 'PARTITION': OrderedDict({'level': 2, 'recursive': True}),
                                 'QUICKSORT': OrderedDict({'level': 3, 'recursive': True})}.items()))

        for i, key in enumerate(sorted(list(self.programs_library.keys()))):
            self.programs_library[key]['index'] = i

        self.prog_to_func = OrderedDict(sorted({'STOP': self._stop,
                             'PTR_1_LEFT': self._ptr_1_left,
                             'PTR_2_LEFT': self._ptr_2_left,
                             'PTR_3_LEFT': self._ptr_3_left,
                             'PTR_1_RIGHT': self._ptr_1_right,
                             'PTR_2_RIGHT': self._ptr_2_right,
                             'PTR_3_RIGHT': self._ptr_3_right,
                             'SWAP': self._swap,
                             'SWAP_2': self._swap_2,
                             'SWAP_3': self._swap_3,
                             'PUSH': self._push,
                             'POP': self._pop}.items()))

        self.prog_to_precondition = OrderedDict(sorted({'STOP': self._stop_precondition,
                                     'RSHIFT': self._rshift_precondition,
                                     'LSHIFT': self._lshift_precondition,
                                     'RESET': self._reset_precondition,
                                     'PARTITION': self._partition_precondition,
                                     'PARTITION_UPDATE': self._partition_update_precondition,
                                     'QUICKSORT': self._quicksort_precondition,
                                     'PTR_1_LEFT': self._ptr_1_left_precondition,
                                     'PTR_2_LEFT': self._ptr_2_left_precondition,
                                     'PTR_3_LEFT': self._ptr_3_left_precondition,
                                     'PTR_1_RIGHT': self._ptr_1_right_precondition,
                                     'PTR_2_RIGHT': self._ptr_2_right_precondition,
                                     'PTR_3_RIGHT': self._ptr_3_right_precondition,
                                     'SWAP': self._swap_precondition,
                                     'SWAP_2': self._swap_2_precondition,
                                     'SWAP_3': self._swap_3_precondition,
                                     'PUSH': self._push_precondition,
                                     'POP': self._pop_precondition}.items()))

        self.prog_to_postcondition = OrderedDict(sorted({'RSHIFT': self._rshift_postcondition,
                                     'LSHIFT': self._lshift_postcondition,
                                     'RESET': self._reset_postcondition,
                                     'PARTITION': self._partition_postcondition,
                                     'PARTITION_UPDATE': self._partition_update_postcondition,
                                     'QUICKSORT': self._quicksort_postcondition}.items()))

        super(QuickSortRecursiveListEnv, self).__init__(self.programs_library, self.prog_to_func,
                                               self.prog_to_precondition, self.prog_to_postcondition)

    def _decr_length_right(self):
        assert self._decr_length_right_precondition(), 'precondition not verified'
        if self.end_pos > self.start_pos:
            self.end_pos -= 1

    def _decr_length_right_precondition(self):
        return self.end_pos > self.start_pos and (self.p1_pos < self.end_pos and self.p2_pos < self.end_pos and self.p3_pos < self.end_pos)

    def _decr_length_left(self):
        assert self._decr_length_left_precondition(), 'precondition not verified'
        if self.start_pos < self.end_pos:
            self.start_pos += 1

    def _decr_length_left_precondition(self):
        return self.start_pos < self.end_pos and (self.p1_pos > self.start_pos and self.p2_pos > self.start_pos and self.p3_pos > self.start_pos)

    def _incr_length_left(self):
        assert self._incr_length_left_precondition(), 'precondition not verified'
        if self.start_pos > 0:
            self.start_pos -= 1

    def _incr_length_left_precondition(self):
        return self.start_pos > 0

    def _incr_length_right(self):
        assert self._incr_length_right_precondition(), 'precondition not verified'
        if self.end_pos < self.length-1:
            self.end_pos += 1

    def _incr_length_right_precondition(self):
        return self.end_pos < self.length-1

    def _ptr_1_left(self):
        assert self._ptr_1_left_precondition(), 'precondition not verified'
        if self.p1_pos > self.start_pos:
            self.p1_pos -= 1

    def _ptr_1_left_precondition(self):
        return self.p1_pos > self.start_pos

    def _stop(self):
        assert self._stop_precondition(), 'precondition not verified'
        pass

    def _stop_precondition(self):
        return True

    def _ptr_2_left(self):
        assert self._ptr_2_left_precondition(), 'precondition not verified'
        if self.p2_pos > self.start_pos:
            self.p2_pos -= 1

    def _ptr_2_left_precondition(self):
        return self.p2_pos > self.start_pos

    def _ptr_3_left(self):
        assert self._ptr_3_left_precondition(), 'precondition not verified'
        if self.p3_pos > self.start_pos:
            self.p3_pos -= 1

    def _ptr_3_left_precondition(self):
        return self.p3_pos > self.start_pos

    def _ptr_1_right(self):
        assert self._ptr_1_right_precondition(), 'precondition not verified'
        if self.p1_pos < self.end_pos:
            self.p1_pos += 1

    def _ptr_1_right_precondition(self):
        return self.p1_pos < self.end_pos

    def _ptr_2_right(self):
        assert self._ptr_2_right_precondition(), 'precondition not verified'
        if self.p2_pos < self.end_pos:
            self.p2_pos += 1

    def _ptr_2_right_precondition(self):
        return self.p2_pos < self.end_pos

    def _ptr_3_right(self):
        assert self._ptr_3_right_precondition(), 'precondition not verified'
        if self.p3_pos < self.end_pos:
            self.p3_pos += 1

    def _ptr_3_right_precondition(self):
        return self.p3_pos < self.end_pos

    def _swap(self):
        assert self._swap_precondition(), 'precondition not verified'
        self.scratchpad_ints[[self.p1_pos, self.p2_pos]] = self.scratchpad_ints[[self.p2_pos, self.p1_pos]]

    def _swap_precondition(self):
        return self.p1_pos != self.p2_pos

    def _swap_2(self):
        assert self._swap_2_precondition(), 'precondition not verified'
        self.scratchpad_ints[[self.p2_pos, self.p3_pos]] = self.scratchpad_ints[[self.p3_pos, self.p2_pos]]

    def _swap_2_precondition(self):
        return self.p2_pos != self.p3_pos

    def _swap_3(self):
        assert self._swap_3_precondition(), 'precondition not verified'
        self.scratchpad_ints[[self.p1_pos, self.p3_pos]] = self.scratchpad_ints[[self.p3_pos, self.p1_pos]]

    def _swap_3_precondition(self):
        return self.p1_pos != self.p3_pos

    def _push(self):
        assert self._push_precondition(), 'precondition not verified'
        self.prog_stack.append(self.p3_pos)
        self.prog_stack.append(self.p2_pos)
        self.prog_stack.append(self.p3_pos)
        self.prog_stack.append(self.p1_pos)
        self.prog_stack.append(self.p3_pos)
        self.prog_stack.append(self.p1_pos)

    def _push_precondition(self):
        return True

    def _pop(self):
        self.p1_pos = self.prog_stack.pop()
        self.p2_pos = self.prog_stack.pop()
        self.p3_pos = self.prog_stack.pop()

    def _pop_precondition(self):
        return len(self.prog_stack) >= 3

    def _compswap_precondition(self):
        list_length = self.end_pos - self.start_pos + 1
        bool = list_length > 1
        bool &= self.p1_pos < self.end_pos
        bool &= self.p2_pos == self.p1_pos or self.p2_pos == (self.p1_pos + 1)
        return bool

    def _compswap_postcondition(self, init_state, state):
        new_scratchpad_ints, new_p1_pos, new_p2_pos, new_p3_pos, new_start_pos, new_end_pos, new_prog_stack = init_state
        new_scratchpad_ints = np.copy(new_scratchpad_ints)
        if new_p1_pos == new_p2_pos and new_p2_pos < new_end_pos:
            new_p2_pos += 1
        idx_left = min(new_p1_pos, new_p2_pos)
        idx_right = max(new_p1_pos, new_p2_pos)
        if new_scratchpad_ints[idx_left] > new_scratchpad_ints[idx_right]:
            new_scratchpad_ints[[idx_left, idx_right]] = new_scratchpad_ints[[idx_right, idx_left]]
        new_state = (new_scratchpad_ints, new_p1_pos, new_p2_pos, new_p3_pos, new_start_pos, new_end_pos, new_prog_stack)
        return self.compare_state(state, new_state)

    def _lshift_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_start_pos, init_end_pos, init_prog_stack = init_state
        scratchpad_ints, p1_pos, p2_pos, p3_pos, start_pos, end_pos, prog_stack = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        bool &= init_start_pos == start_pos
        bool &= init_end_pos == end_pos
        if init_p1_pos > init_start_pos:
            bool &= p1_pos == (init_p1_pos-1)
        else:
            bool &= p1_pos == init_p1_pos

        if init_p2_pos > init_start_pos:
            bool &= p2_pos == (init_p2_pos-1)
        else:
            bool &= p2_pos == init_p2_pos

        if init_p3_pos > init_start_pos:
            bool &= p3_pos == (init_p3_pos-1)
        else:
            bool &= p3_pos == init_p3_pos

        return bool

    def _rshift_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_start_pos, init_end_pos, init_prog_stack = init_state
        scratchpad_ints, p1_pos, p2_pos, p3_pos, start_pos, end_pos, prog_stack = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        bool &= init_start_pos == start_pos
        bool &= init_end_pos == end_pos
        if init_p1_pos < init_end_pos:
            bool &= p1_pos == (init_p1_pos+1)
        else:
            bool &= p1_pos == init_p1_pos

        if init_p2_pos < init_end_pos:
            bool &= p2_pos == (init_p2_pos+1)
        else:
            bool &= p2_pos == init_p2_pos

        if init_p3_pos < init_end_pos:
            bool &= p3_pos == (init_p3_pos+1)
        else:
            bool &= p3_pos == init_p3_pos

        return bool

    def _reset_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_start_pos, init_end_pos, prog_stack = init_state
        scratchpad_ints, p1_pos, p2_pos, p3_pos, start_pos, end_pos, prog_stack = state
        bool = np.array_equal(init_scratchpad_ints, scratchpad_ints)
        #bool &= init_start_pos == start_pos
        #bool &= init_end_pos == end_pos
        bool &= (p1_pos == init_start_pos and p2_pos == init_start_pos and p3_pos == init_start_pos)
        return bool

    def _quicksort_postcondition(self, init_state, state):
        init_scratchpad_ints, init_p1_pos, init_p2_pos, init_p3_pos, init_start_pos, init_end_pos, init_prog_stack = init_state
        scratchpad_ints, p1_pos, p2_pos, p3_pos, start_pos, end_pos, prog_stack = state
        # check if list is sorted
        bool &= np.all(scratchpad_ints[:end_pos] <= scratchpad_ints[(start_pos+1):(end_pos+1)])
        bool &= (len(prog_stack) == 0)
        return bool

    def _partition_update_function(self, arr, low, high, j):
        if arr[j] < arr[high]:
            arr[[low, j]] = arr[[j, low]]
            return arr, low+1, high, j+1
        else:
            return arr, low, high, j+1

    def _partition_function(self, arr, low, high, j):
        if j <= high:
            arr, low, high, j = self._partition_update_function(arr, low, high, j)
            return self._partition_function(arr, low, high, j)
        else:
            arr[[low, high]] = arr[[high, low]]
            return arr, low, high, j;

    def _quicksort(arr, low, high, pivot):
        if low < high:
            arr, l, h, j = self._partition_function(arr, low, high, pivot)
            # PUSH(p3, p2, p3, p1, p3, p1)
            # POP(p1, p3, p1)
            # RSHIFT(p2)
            arr = self.quicksort(arr, low, l-1, low)
            # POP(p3, p2, p3)
            return self.quicksort(arr, l+1, high, l+1)
        return arr

    def _partition_update_postcondition(self, init_state, state):
        # Generate the output it should have
        new_scratchpad_ints, new_p1_pos, new_p2_pos, new_p3_pos, new_start_pos, new_end_pos, new_prog_stack = init_state
        new_scratchpad_ints, low, high, j = self._partition_update_function(np.copy(new_scratchpad_ints), new_p1_pos, new_p2_pos, new_p3_pos)
        new_p1_pos = low
        new_p2_pos = high
        new_p3_pos = j

        # We create a new state and then we compare
        new_state = (new_scratchpad_ints, new_p1_pos, new_p2_pos, new_p3_pos, new_start_pos, new_end_pos, new_prog_stack)
        return self.compare_state(state, new_state)

    def _partition_postcondition(self, init_state, state):
        new_scratchpad_ints, new_p1_pos, new_p2_pos, new_p3_pos, new_start_pos, new_end_pos, new_prog_stack = init_state
        new_scratchpad_ints, low, high, j = self._partition_function(np.copy(new_scratchpad_ints), new_p1_pos, new_p2_pos, new_p3_pos)
        new_p1_pos = low
        new_p2_pos = high
        new_p3_pos = j
        new_state = (new_scratchpad_ints, new_p1_pos, new_p2_pos, new_p3_pos, new_start_pos, new_end_pos, new_prog_stack)
        return self.compare_state(state, new_state)

    def _lshift_precondition(self):
        return self.p1_pos > self.start_pos or self.p2_pos > self.start_pos or self.p3_pos > self.start_pos

    def _rshift_precondition(self):
        return self.p1_pos < self.end_pos or self.p2_pos < self.end_pos or self.p3_pos < self.end_pos

    def _partition_precondition(self):
        return self.p1_pos >= 0 and self.p1_pos <= self.p3_pos and self.p3_pos <= self.p2_pos+1 and self.p2_pos <= self.length-1

    def _partition_update_precondition(self):
        return self.p3_pos <= self.p2_pos and self.p3_pos >= self.p1_pos and self.p1_pos <= self.p2_pos

    def _reset_precondition(self):
        bool = (self.p1_pos > self.start_pos or self.p2_pos > self.start_pos or self.p3_pos > self.start_pos)
        #reset_index = self.programs_library['RESET']['index']
        #if self.current_task_index == reset_index:
        #    bool &= self._decr_length_right_precondition()
        return bool

    def _quicksort_precondition(self):
        return self.p1_pos >= 0 and self.p1_pos <= self.p2_pos and self.p2_pos <= self.length-1

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
          a: numpy array representing the 10-hot-encoding of a digit.
          one_encoding:

        Returns:
          the digit encoded in one_encoding

        """
        return np.argmax(one_encoding)

    def reset_env(self):
        """Reset the environment. The list are values are draw randomly. The pointers are initialized at position 0
        (at left position of the list).

        """
        assert self.length > 1, "list length must be greater than 1"
        self.start_pos = 0
        self.end_pos = self.length-1
        self.scratchpad_ints = np.random.randint(10, size=self.length)
        current_task_name = self.get_program_from_index(self.current_task_index)
        if current_task_name == 'BUBBLE' or current_task_name == 'BUBBLESORT':
            init_pointers_pos1 = 0
            init_pointers_pos2 = 0
            init_pointers_pos3 = 0
        elif current_task_name == 'QUICKSORT':
            init_pointers_pos1 = 0
            init_pointers_pos2 = self.end_pos
            init_pointers_pos3 = 0
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
                if not (init_pointers_pos1 == 0 and init_pointers_pos2 == 0 and init_pointers_pos3 == 0):
                    break
        elif current_task_name == 'COMPSWAP':
            init_pointers_pos1 = int(np.random.randint(0, self.length - 1))
            init_pointers_pos2 = int(np.random.choice([init_pointers_pos1, init_pointers_pos1 + 1]))
            init_pointers_pos3 = int(np.random.randint(0, self.length - 1))
        elif current_task_name == 'PARTITION':
            while True:
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                if not init_pointers_pos2 == 0:
                    break
            init_pointers_pos1 = int(np.random.randint(0, init_pointers_pos2))
            init_pointers_pos3 = init_pointers_pos1
        elif current_task_name == 'PARTITION_UPDATE':
            while True:
                init_pointers_pos2 = int(np.random.randint(0, self.length))
                if not init_pointers_pos2 == 0:
                    break
            init_pointers_pos1 = int(np.random.randint(0, init_pointers_pos2))
            init_pointers_pos3 = init_pointers_pos1
        else:

            raise NotImplementedError('Unable to reset env for this program...')

        self.p1_pos = init_pointers_pos1
        self.p2_pos = init_pointers_pos2
        self.p3_pos = init_pointers_pos3
        self.prog_stack = []
        self.has_been_reset = True

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
        pointers_same_pos = int(self.p1_pos == self.p2_pos)
        pointers2_same_pos = int(self.p2_pos == self.p3_pos)
        pointers3_same_pos = int(self.p1_pos == self.p3_pos)
        pt_1_left = int(self.p1_pos == self.start_pos)
        pt_2_left = int(self.p2_pos == self.start_pos)
        pt_3_left = int(self.p3_pos == self.start_pos)
        pt_1_right = int(self.p1_pos == self.end_pos)
        pt_2_right = int(self.p2_pos == self.end_pos)
        pt_3_right = int(self.p3_pos == self.end_pos)
        p1p2p3 = np.eye(10)[[p1_val, p2_val, p3_val]].reshape(-1)  # one hot encoding of values at pointers pos
        is_stack_empty = int(len(self.prog_stack) > 0)
        #TODO: fix this and the get_observaion_dim method below.
        bools = np.array([
            pt_1_left,
            pt_1_right,
            pt_2_left,
            pt_2_right,
            pt_3_left,
            pt_3_right,
            pointers_same_pos,
            pointers2_same_pos,
            pointers3_same_pos,
            is_sorted,
            is_stack_empty
        ])
        return np.concatenate((p1p2p3, bools), axis=0)

    def get_observation_dim(self):
        """

        Returns:
            the size of the observation tensor
        """
        return 3 * 10 + 11
        #return 3 * 10 + len(self.prog_stack)*10 + 9

    def get_state(self):
        """Returns the current state.

        Returns:
            the environment state

        """
        assert self.has_been_reset, 'Need to reset the environment before getting states'
        return np.copy(self.scratchpad_ints), self.p1_pos, self.p2_pos, self.p3_pos, self.start_pos, self.end_pos, self.prog_stack.copy()

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
        self.start_pos = state[4]
        self.end_pos = state[5]
        self.prog_stack = state[6].copy()

    def get_state_str(self, state):
        """Print a graphical representation of the environment state"""
        scratchpad = state[0].copy()
        p1_pos = state[1]
        p2_pos = state[2]
        p3_pos = state[3]
        start_pos = state[4]
        end_pos = state[5]
        scratchpad = scratchpad[start_pos:end_pos+1]
        prog_stack = state[6]
        str = 'list: {}, p1 : {}, p2 : {}, p3 : {}, start_pos: {}, end_pos: {}, prog_stack: {}'.format(scratchpad, p1_pos,
                                                                              p2_pos, p3_pos, start_pos, end_pos, prog_stack)
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
        bool = np.array_equal(state1[0], state2[0])
        bool &= (state1[1] == state2[1])
        bool &= (state1[2] == state2[2])
        bool &= (state1[3] == state2[3])
        bool &= (state1[4] == state2[4])
        bool &= (state1[5] == state2[5])
        bool &= (state1[6] == state2[6])
        return bool

    def _is_sorted(self):
        """Assert is the list is sorted or not.

        Args:

        Returns:
            True if the list is sorted, False otherwise

        """
        arr = self.scratchpad_ints
        return np.all(arr[:self.length-1] <= arr[1:self.length])

    def start_task(self, task_index):
        if self.tasks_list.count(task_index) > 0:
            task = self.get_program_from_index(task_index)
            if task == 'RESET':
                pass
                #self._decr_length_right()
            if task == 'QUICKSORT':
                pass
            if task == 'PARTITION':
                pass
        return super(QuickSortRecursiveListEnv, self).start_task(task_index)

    def end_task(self):
        current_task = self.get_program_from_index(self.current_task_index)
        if current_task == 'RESET':
            #if self.tasks_list.count(self.current_task_index) > 1:
                #self._incr_length_right()
            pass
        if current_task == 'QUICKSORT':
            pass
        if current_task == 'PARTITION':
            pass
            #if self.tasks_list.count(self.current_task_index) > 1:
            #    self._incr_length_left()
        super(QuickSortRecursiveListEnv, self).end_task()
