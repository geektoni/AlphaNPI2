from environments.recursive_list_env import RecursiveListEnv, RecursiveListEnvEncoder
from core.curriculum import CurriculumScheduler
from core.policy import Policy
import core.config as conf
from core.trainer import Trainer
from core.prioritized_replay_buffer import PrioritizedReplayBuffer
import torch
import numpy as np
import argparse
import time
from tensorboardX import SummaryWriter

if __name__ == "__main__":

    # Get command line params
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help='random seed', default=1, type=int)
    parser.add_argument('--tensorboard', help='display on tensorboard', action='store_true')
    parser.add_argument('--verbose', help='print training monitoring in console', action='store_true')
    parser.add_argument('--save-model', help='save neural network model', action='store_true')
    parser.add_argument('--save-results', help='save training progress in .txt file', action='store_true')
    parser.add_argument('--num-cpus', help='number of cpus to use', default=8, type=int)
    args = parser.parse_args()

    # Get arguments
    seed = args.seed
    tensorboard = args.tensorboard
    verbose = args.verbose
    save_model = args.save_model
    save_results = args.save_results
    num_cpus = args.num_cpus

    # Set number of cpus used
    torch.set_num_threads(num_cpus)

    # get date and time
    ts = time.localtime(time.time())
    date_time = '{}_{}_{}-{}_{}_{}'.format(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])
    # Path to save policy
    model_save_path = '../models/recursive_list_npi_{}-{}.pth'.format(date_time, seed)
    # Path to save results
    results_save_path = '../results/recursive_list_npi_{}-{}.txt'.format(date_time, seed)
    # Path to tensorboard
    tensorboard_path = 'runs/recursive_list_npi_{}-{}'.format(date_time, seed)

    # Instantiate tensorboard writer
    if tensorboard:
        writer = SummaryWriter(tensorboard_path)

    # Instantiate file writer
    if save_results:
        results_file = open(results_save_path, 'w')

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load environment constants
    env_tmp = RecursiveListEnv(length=5, encoding_dim=conf.encoding_dim)
    num_programs = env_tmp.get_num_programs()
    num_non_primary_programs = env_tmp.get_num_non_primary_programs()
    observation_dim = env_tmp.get_observation_dim()
    programs_library = env_tmp.programs_library

    # Load Alpha-NPI policy
    encoder = RecursiveListEnvEncoder(env_tmp.get_observation_dim(), conf.encoding_dim)
    indices_non_primary_programs = [p['index'] for _, p in programs_library.items() if p['level'] > 0]
    policy = Policy(encoder, conf.hidden_size, num_programs, num_non_primary_programs, conf.program_embedding_dim,
                    conf.encoding_dim, indices_non_primary_programs, conf.learning_rate)

    # Load replay buffer
    idx_tasks = [prog['index'] for key, prog in env_tmp.programs_library.items() if prog['level'] > 0]
    buffer = PrioritizedReplayBuffer(conf.buffer_max_length, idx_tasks, p1=conf.proba_replay_buffer)

    # Prepare mcts params
    max_depth_dict = {1: 5, 2: 5, 3: 5}
    mcts_train_params = {'number_of_simulations': conf.number_of_simulations, 'max_depth_dict': max_depth_dict,
                         'temperature': conf.temperature, 'c_puct': conf.c_puct, 'exploit': False,
                         'level_closeness_coeff': conf.level_closeness_coeff, 'gamma': conf.gamma,
                         'use_dirichlet_noise': True, 'max_recursion_depth': 7*2+1}

    mcts_test_params = {'number_of_simulations': conf.number_of_simulations_for_validation,
                        'max_depth_dict': max_depth_dict, 'temperature': conf.temperature,
                        'c_puct': conf.c_puct, 'exploit': True, 'level_closeness_coeff': conf.level_closeness_coeff,
                        'gamma': conf.gamma, 'max_recursion_depth': 7*2+1}

    # Load curriculum sequencer
    curriculum_scheduler = CurriculumScheduler(conf.reward_threshold, num_non_primary_programs, programs_library,
                                               moving_average=0.99)

    # Instanciate trainer
    trainer = Trainer(env_tmp, policy, buffer, curriculum_scheduler, mcts_train_params,
                      mcts_test_params, conf.num_validation_episodes, conf.num_episodes_per_task, conf.batch_size,
                      conf.num_updates_per_episode, verbose)

    min_length = 2
    max_length = 4
    validation_length = 7
    # Start training
    for iteration in range(conf.num_iterations):
        # play one iteration
        task_index = curriculum_scheduler.get_next_task_index()
        task_level = env_tmp.get_program_level_from_index(task_index)
        length = np.random.randint(min_length, max_length+1)

        env = RecursiveListEnv(length=length, encoding_dim=conf.encoding_dim)
        trainer.env = env
        trainer.play_iteration(task_index)

        # perform validation
        if verbose:
            print("Start validation .....")
        for idx in curriculum_scheduler.get_tasks_of_maximum_level():
            task_level = env_tmp.get_program_level_from_index(idx)
            length = validation_length
            env = RecursiveListEnv(length=length, encoding_dim=conf.encoding_dim)
            trainer.env = env
            # Evaluate performance on task idx
            v_rewards, v_lengths, programs_failed_indices = trainer.perform_validation_step(idx)
            # Update curriculum statistics
            curriculum_scheduler.update_statistics(idx, v_rewards)

        # display training progress in tensorboard
        if tensorboard:
            for idx in curriculum_scheduler.get_tasks_of_maximum_level():
                v_task_name = env.get_program_from_index(idx)
                # record on tensorboard
                writer.add_scalar('validation/' + v_task_name, curriculum_scheduler.get_statistic(idx), iteration)

        # write training progress in txt file
        if save_results:
            str = 'Iteration: {}'.format(iteration)
            for idx in curriculum_scheduler.indices_non_primary_programs:
                task_name = env.get_program_from_index(idx)
                str += ', %s:%.3f' % (task_name, curriculum_scheduler.get_statistic(idx))
            str += '\n'
            results_file.write(str)

        # print new training statistics
        if verbose:
            curriculum_scheduler.print_statistics()
            print('')
            print('')

        # If succeed on al tasks, stop training
        if curriculum_scheduler.maximum_level > env.get_maximum_level():
            break

        # Save policy
        if save_model:
            torch.save(policy.state_dict(), model_save_path)

    # Close tensorboard writer
    if verbose:
        print('End of training !')
    if tensorboard:
        writer.close()
    if save_results:
        results_file.close()
