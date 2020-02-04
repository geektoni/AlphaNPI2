from environments.quicksort_list_env import ListEnvEncoder, QuickSortListEnv
from core.curriculum import CurriculumScheduler
from core.policy import Policy
import core.config as conf
from core.trainer import Trainer
from core.prioritized_replay_buffer import PrioritizedReplayBuffer
import argparse
import numpy as np
import torch
from tensorboardX import SummaryWriter
import time

if __name__ == "__main__":

    # Get command line params
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help='random seed', default=1, type=int)
    parser.add_argument('--tensorboard', help='display on tensorboard', action='store_true')
    parser.add_argument('--verbose', help='print training monitoring in console', action='store_true')
    parser.add_argument('--save-model', help='save neural network model', action='store_true')
    parser.add_argument('--save-results', help='save training progress in .txt file', action='store_true')
    parser.add_argument('--num-cpus', help='number of cpus to use', default=8, type=int)
    parser.add_argument('--load-model', help='Load a pretrained model and train from there', default="", type=str)
    parser.add_argument('--min-length', help='Minimum size of the list we want to order', default=2, type=int)
    parser.add_argument('--max-length', help='Max size of the list we want to order', default=7, type=int)
    parser.add_argument('--validation-length', help='Size of the validation lists we want to order', default=7, type=int)
    parser.add_argument('--start-level', help='Specify up to which level we are trying to learn', default=1, type=int)
    parser.add_argument('--tb-base-dir', help='Specify base tensorboard dir', default="runs", type=str)
    parser.add_argument('--structural-constraint', help="Use the structural constraint to train", action='store_true')
    parser.add_argument('--gamma', help="Specify gamma discount factor", default=0.97, type=float)
    parser.add_argument('--penalize-level-0', help="Penalize level 0 operations when computing the Q-value", default=True, action='store_false')
    parser.add_argument('--level-0-penalty', help="Custom penalty value for the level 0 actions", default=1.0, type=float)
    parser.add_argument('--expose-stack', help="When observing the environment, simply expose the firs two element of the stack", default=False, action='store_true')
    parser.add_argument('--sample-error-prob', help="Probability of sampling error envs when doing training", default=0.3, type=float)
    parser.add_argument('--without-partition-update', help="Train everything without the partition update program", default=False, action="store_true")
    parser.add_argument('--reduced-operation-set', help="Train everything with a reduced set of operations", default=False, action="store_true")
    parser.add_argument('--keep-training', help="Keep training even if we reach 'perfection' on all the task", default=False, action="store_true")
    parser.add_argument('--recursive-quicksort', help="The QUICKSORT_UPDATE function is made recursive.", default=False, action="store_true")
    args = parser.parse_args()

    # Get arguments
    seed = args.seed
    tensorboard = args.tensorboard
    base_tb_dir = args.tb_base_dir
    verbose = args.verbose
    save_model = args.save_model
    save_results = args.save_results
    num_cpus = args.num_cpus
    sample_error_prob = args.sample_error_prob
    conf.gamma = args.gamma
    conf.penalize_level_0 = args.penalize_level_0
    conf.level_0_penalty = args.level_0_penalty

    # Verbose output
    if verbose:
        print(args)

    load_model = False
    if args.load_model != "":
        load_model = True

    custom_start_level = False
    if args.start_level != 0:
        custom_start_level = True

    # Set if we are using the structural constraint
    if args.structural_constraint:
        conf.structural_constraint = True

    # Set number of cpus used
    torch.set_num_threads(num_cpus)

    # get date and time
    ts = time.localtime(time.time())
    date_time = '{}_{}_{}-{}_{}_{}'.format(ts[0], ts[1], ts[2], ts[3], ts[4], ts[5])
    # Path to save policy
    model_save_path = '../models/list_npi_{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.pth'.format(date_time, seed, args.structural_constraint,
                                                               args.penalize_level_0, args.level_0_penalty, args.expose_stack,
                                                                           sample_error_prob, args.without_partition_update,
                                                                                    args.reduced_operation_set,
                                                                                    args.keep_training,
                                                                                    args.recursive_quicksort)
    # Path to save results
    results_save_path = '../results/list_npi_{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.txt'.format(date_time, seed, args.structural_constraint,
                                                               args.penalize_level_0, args.level_0_penalty, args.expose_stack,
                                                                              sample_error_prob, args.without_partition_update,
                                                                                       args.reduced_operation_set,
                                                                                       args.keep_training,
                                                                                       args.recursive_quicksort)
    # Path to tensorboard
    tensorboard_path = '{}/list_npi_{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(base_tb_dir, date_time, seed, args.structural_constraint,
                                                               args.penalize_level_0, args.level_0_penalty, args.expose_stack,
                                                                 sample_error_prob, args.without_partition_update,
                                                                          args.reduced_operation_set,
                                                                          args.keep_training,
                                                                          args.recursive_quicksort)

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
    env_tmp = QuickSortListEnv(length=5, encoding_dim=conf.encoding_dim, expose_stack=args.expose_stack,
                               sample_from_errors_prob=sample_error_prob,
                               without_partition_update=args.without_partition_update,
                               reduced_set=args.reduced_operation_set,
                               recursive_version=args.recursive_quicksort)
    num_programs = env_tmp.get_num_programs()
    num_non_primary_programs = env_tmp.get_num_non_primary_programs()
    observation_dim = env_tmp.get_observation_dim()
    programs_library = env_tmp.programs_library

    # Load alphanpi policy
    encoder = ListEnvEncoder(env_tmp.get_observation_dim(), conf.encoding_dim)
    indices_non_primary_programs = [p['index'] for _, p in programs_library.items() if p['level'] > 0]
    policy = Policy(encoder, conf.hidden_size, num_programs, num_non_primary_programs, conf.program_embedding_dim,
                    conf.encoding_dim, indices_non_primary_programs, conf.learning_rate)

    # Load a pre-trained policy (to speed up testing)
    if load_model:
        policy.load_state_dict(torch.load(args.load_model))

    # Load replay buffer
    idx_tasks = [prog['index'] for key, prog in env_tmp.programs_library.items() if prog['level'] > 0]
    buffer = PrioritizedReplayBuffer(conf.buffer_max_length, idx_tasks, p1=conf.proba_replay_buffer)

    # Load curriculum sequencer
    curriculum_scheduler = CurriculumScheduler(conf.reward_threshold, num_non_primary_programs, programs_library,
                                               moving_average=0.99)
    curriculum_scheduler.maximum_level = args.start_level

    # Prepare mcts params
    length = 5

    if args.without_partition_update:
        max_depth_dict = {1: 3 * (length - 1) + 2, 2: 4, 3: 4, 4: length + 2}
    elif args.reduced_operation_set:
        max_depth_dict = {1: 3 * (length - 1) + 2, 2: 6, 3: length + 2}
    else:
        max_depth_dict = {1: 3, 2: 2 * (length - 1) + 2, 3: 4, 4: 4, 5: length + 2}

    mcts_train_params = {'number_of_simulations': conf.number_of_simulations, 'max_depth_dict': max_depth_dict,
                         'temperature': conf.temperature, 'c_puct': conf.c_puct, 'exploit': False,
                         'level_closeness_coeff': conf.level_closeness_coeff, 'gamma': conf.gamma,
                         'use_dirichlet_noise': True, 'use_structural_constraint': conf.structural_constraint,
                         'penalize_level_0': conf.penalize_level_0, 'level_0_penalty': conf.level_0_custom_penalty,
                         'max_recursion_depth': length}

    mcts_test_params = {'number_of_simulations': conf.number_of_simulations_for_validation,
                        'max_depth_dict': max_depth_dict, 'temperature': conf.temperature,
                        'c_puct': conf.c_puct, 'exploit': True, 'level_closeness_coeff': conf.level_closeness_coeff,
                        'gamma': conf.gamma, 'use_structural_constraint': conf.structural_constraint,
                        'penalize_level_0': conf.penalize_level_0, 'level_0_penalty': conf.level_0_custom_penalty,
                        'max_recursion_depth': length}

    # Specify a custom start level
    if custom_start_level:
        curriculum_scheduler.maximum_level = args.start_level

    # Instanciate trainer
    trainer = Trainer(env_tmp, policy, buffer, curriculum_scheduler, mcts_train_params,
                      mcts_test_params, conf.num_validation_episodes, conf.num_episodes_per_task, conf.batch_size,
                      conf.num_updates_per_episode, verbose)

    min_length = args.min_length
    max_length = args.max_length
    validation_length = args.validation_length
    failed_executions_envs = None

    # Start training
    for iteration in range(conf.num_iterations):
        # play one iteration
        task_index = curriculum_scheduler.get_next_task_index()
        task_level = env_tmp.get_program_level_from_index(task_index)
        length = np.random.randint(min_length, max_length+1)
        env = QuickSortListEnv(length=length, encoding_dim=conf.encoding_dim, expose_stack=args.expose_stack,
                               sample_from_errors_prob=sample_error_prob,
                               without_partition_update=args.without_partition_update,
                               reduced_set=args.reduced_operation_set,
                               recursive_version=args.recursive_quicksort)

        if args.without_partition_update:
            max_depth_dict = {1: 3 * (length - 1) + 2, 2: 4, 3: 4, 4: length + 2}
        elif args.reduced_operation_set:
            max_depth_dict = {1: 3 * (length - 1) + 2, 2: 6, 3: length + 2}
        elif args.recursive_quicksort:
            max_depth_dict =  {1: 3, 2: 2 * (length - 1) + 2, 3: 4, 4: 4, 5: length + 1, 6: 3}
        else:
            max_depth_dict = {1: 3, 2: 2 * (length - 1) + 2, 3: 4, 4: 4, 5: length + 2}

        # Restore the previous failed executions
        if failed_executions_envs != None:
            env.failed_executions_env = failed_executions_envs

        trainer.env = env
        trainer.mcts_train_params['max_depth_dict'] = max_depth_dict
        trainer.mcts_test_params['max_depth_dict'] = max_depth_dict
        trainer.play_iteration(task_index)

        # Save the failed execution env
        failed_executions_envs = env.failed_executions_env

        # perform validation
        if verbose:
            print("Start validation .....")
        for idx in curriculum_scheduler.get_tasks_of_maximum_level():
            task_level = env_tmp.get_program_level_from_index(idx)
            length = validation_length
            env = QuickSortListEnv(length=length, encoding_dim=conf.encoding_dim, expose_stack=args.expose_stack,
                                   validation_mode=True,
                                   without_partition_update=args.without_partition_update,
                                   reduced_set=args.reduced_operation_set,
                                   recursive_version=args.recursive_quicksort)

            if args.without_partition_update:
                max_depth_dict = {1: 3 * (length - 1) + 2, 2: 4, 3: 4, 4: length + 2}
            elif args.reduced_operation_set:
                max_depth_dict = {1: 3 * (length - 1) + 2, 2: 6, 3: length + 2}
            elif args.recursive_quicksort:
                max_depth_dict = {1: 3, 2: 2 * (length - 1) + 2, 3: 4, 4: 4, 5: length + 1, 6: 3}
            else:
                max_depth_dict = {1: 3, 2: 2 * (length - 1) + 2, 3: 4, 4: 4, 5: length + 2}

            trainer.env = env
            trainer.mcts_train_params['max_depth_dict'] = max_depth_dict
            trainer.mcts_test_params['max_depth_dict'] = max_depth_dict
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

        # If succeed on al tasks, go directly to next list length
        if curriculum_scheduler.maximum_level > env.get_maximum_level():
            if not args.keep_training:
                break
            else:
                # keep on training
                curriculum_scheduler.maximum_level = env.get_maximum_level()

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
