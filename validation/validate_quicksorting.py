from environments.quicksort_list_env import QuickSortListEnv, ListEnvEncoder
from core.policy import Policy
from core.network_only import NetworkOnly
import core.config as conf
import torch
import argparse
from core.mcts import MCTS
import numpy as np
import time

from tqdm import tqdm

import os

if __name__ == "__main__":

    # Path to load policy
    #default_load_path = '../models/list_npi_2019_5_16-10_19_59-1.pth'
    default_load_path = '../models/list_npi_2019_5_13-9_26_38-1.pth'

    # Get command line params
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help='random seed', default=1, type=int)
    parser.add_argument("--load-path", help='path to model to validate', default=default_load_path)
    parser.add_argument('--verbose', help='print training monitoring in console', action='store_true')
    parser.add_argument('--save-results', help='save training progress in .txt file', action='store_true')
    parser.add_argument('--num-cpus', help='number of cpus to use', default=8, type=int)
    parser.add_argument('--min-length', help='Minimum size of the list we want to order', default=2, type=int)
    parser.add_argument('--max-length', help='Max size of the list we want to order', default=7, type=int)
    parser.add_argument('--validation-length', help='Size of the validation lists we want to order', default=7, type=int)
    parser.add_argument('--operation', help="Operation we want to test", default="QUICKSORT", type=str)
    args = parser.parse_args()

    # Get arguments
    seed = args.seed
    verbose = args.verbose
    save_results = args.save_results
    load_path = args.load_path
    num_cpus = args.num_cpus

    # Set number of cpus used
    torch.set_num_threads(num_cpus)

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Obtain the various configuration options from the name.
    filename = os.path.split(load_path)[1]
    values = filename.split("-")

    date = values[0]
    time_ = values[1]
    seed = values[2]
    str_c = values[3].lower() == "true"
    pen_level_0 = values[4].lower() == "true"
    leve_0_pen = float(values[5])
    expose_stack = values[6].lower() == "true"
    samp_err_poss = float(values[7])
    without_p_upd = values[8].lower() == "true"
    reduced_op_set = values[9].lower() == "true"
    keep_training = values[10].split(".")[0].lower() == "true"


    if save_results:
        # get date and time
        ts = time.localtime(time.time())
        date_time = '{}_{}_{}-{}_{}_{}'.format(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])
        results_save_path = '../results/validation_list_npi_{}-{}-{}-{}-{}-{}.txt'.format(date_time, args.operation, samp_err_poss, reduced_op_set, without_p_upd, seed)
        results_file = open(results_save_path, 'w')

    # Load environment constants
    env_tmp = QuickSortListEnv(length=5, encoding_dim=conf.encoding_dim, expose_stack=expose_stack,
                               without_partition_update=without_p_upd, sample_from_errors_prob=samp_err_poss,
                               reduced_set=reduced_op_set)
    num_programs = env_tmp.get_num_programs()
    num_non_primary_programs = env_tmp.get_num_non_primary_programs()
    observation_dim = env_tmp.get_observation_dim()
    programs_library = env_tmp.programs_library

    # Load Alpha-NPI policy
    encoder = ListEnvEncoder(env_tmp.get_observation_dim(), conf.encoding_dim)
    indices_non_primary_programs = [p['index'] for _, p in programs_library.items() if p['level'] > 0]
    policy = Policy(encoder, conf.hidden_size, num_programs, num_non_primary_programs, conf.program_embedding_dim,
                    conf.encoding_dim, indices_non_primary_programs, conf.learning_rate)

    policy.load_state_dict(torch.load(load_path))


    # Start validation
    if verbose:
        print('Start validation for model: {}'.format(load_path))

    if save_results:
        results_file.write('Validation on model: {}'.format(load_path) + ' \n')

    for len in [5, 20, 60, 100]:

        print("** Start validation for len = {}".format(len))

        mcts_rewards_normalized = []
        mcts_rewards = []
        network_only_rewards = []

        if without_p_upd:
            max_depth_dict = {1: 3 * (len - 1) + 2, 2: 4, 3: 4, 4: len + 2}
        elif reduced_op_set:
            max_depth_dict = {1: 3 * (len - 1) + 2, 2: 6, 3: len + 2}
        else:
            max_depth_dict = {1: 3, 2: 2 * (len - 1) + 2, 3: 4, 4: 4, 5: len + 2}

        mcts_test_params = {'number_of_simulations': conf.number_of_simulations_for_validation,
                            'max_depth_dict': max_depth_dict, 'temperature': conf.temperature,
                            'c_puct': conf.c_puct, 'exploit': True, 'level_closeness_coeff': conf.level_closeness_coeff,
                            'gamma': conf.gamma, "penalize_level_0": pen_level_0, 'use_structural_constraint': str_c,
                            'verbose': False}

        for _ in tqdm(range(100)):

            env = QuickSortListEnv(length=len, encoding_dim=conf.encoding_dim,
                                   expose_stack=expose_stack, without_partition_update=without_p_upd,
                                   sample_from_errors_prob=samp_err_poss, reduced_set=reduced_op_set)
            operation_index = env.programs_library[args.operation]['index']

            # Test with mcts
            mcts = MCTS(policy, env, operation_index, **mcts_test_params)
            res = mcts.sample_execution_trace()
            mcts_reward = res[7]
            mcts_rewards.append(mcts_reward)
            if mcts_reward > 0:
                mcts_rewards_normalized.append(1.0)
            else:
                mcts_rewards_normalized.append(0.0)

            # Test with network alone
            network_only = NetworkOnly(policy, env, max_depth_dict)
            netonly_reward, _ = network_only.play(operation_index)
            network_only_rewards.append(netonly_reward)

        mcts_rewards_normalized_mean = np.mean(np.array(mcts_rewards_normalized))
        mcts_rewards_mean = np.mean(np.array(mcts_rewards))
        network_only_rewards_mean = np.mean(np.array(network_only_rewards))

        if verbose:
            print('Length: {}, mcts mean reward: {}, mcts mean normalized reward: {}, '
                  'network only mean reward: {}'.format(len, mcts_rewards_mean, mcts_rewards_normalized_mean,
                                                        network_only_rewards_mean))

        if save_results:
            str = 'Length: {}, mcts mean reward: {}, mcts mean normalized reward: {}, ' \
                  'network only mean reward: {}'.format(len, mcts_rewards_mean, mcts_rewards_normalized_mean,
                                                        network_only_rewards_mean)
            results_file.write(str + ' \n')

    if save_results:
        results_file.close()
