import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from src.logger import get_logger
from src.utils import print_device_info, get_device, make_env
import src.agents as Agents
from src.cp.cp_solver import CPJobShopSolver
from src.experiments.dispatching_rules_wandb import execute_fifo_worker, execute_mwkr_worker
from src.tuning.hyperparameter_search import run_sweep
from src.experiments.train_ppo_multi_env import train_agent_multi_env


class CustomParser():
    def __init__(self):
        self.main_parser = argparse.ArgumentParser(
            description='Permutation Job Shop Scheduler',
            usage='''main <command> [<args>]

            The most commonly used commands are:
            reinforcement-learning      Execute reinforcement-learning on envs.
            supervised-learning         Execute supervised-learning on datasets.         
        ''')
        
        self.subparsers = self.main_parser.add_subparsers(title='subcommands', 
            dest='command', help="Subcommand to run.")
        ######################################
        #      Add your subparsers here      #
        ######################################
        self._add_supervised_parser()
        self._add_reinforcement_parser()
        self._add_constraint_programming_parser()
        self._add_dispatching_rules_parser()
        self._add_reinforcement_tuning_parser()

        ######################################

        self.args = self.main_parser.parse_args()

    def _add_dispatching_rules_parser(self):
        # create the parser for the "supervised-learning" subcommand
        self.parser_dr_command = self.subparsers.add_parser('dr', 
            help='Dispatching rules help.')
        self.parser_dr_command.add_argument('--input_files', metavar='FILE', nargs='+', 
            help='Input files to process')
        self.parser_dr_command.add_argument('--dispatching_rule', type=str.lower, 
            action="store", choices=["fifo, mwkr, all"], default="all", 
            help='Which dispatching-rule to choose.')
        

    def _add_supervised_parser(self):
        # create the parser for the "supervised-learning" subcommand
        self.parser_sl_command = self.subparsers.add_parser('sl', 
            help='Supervised-learning help.')
        self.parser_sl_command.add_argument('--epochs', type=int, default=20, 
            help='Number of epochs to train in supervised-learning mode. Default is 20.')
        self.parser_sl_command.add_argument('--dropout', type=float, default=0.5, 
            help='Probability of an element in the network to be zeroed. Default: 0.5')
        self.parser_sl_command.add_argument('--learning_rate', type=float, default=0.0001, 
            help='Learning rate specified network. Default is 0.0001.')
        self.parser_sl_command.add_argument('--batch_size', type=int, default=128, 
            help='Batch size for training. Default is 128.')
        transpose_choices = [f"transpose-{x}" for x in range(1,16)]
        dataset_choices = (["no-permutation", "random"])
        dataset_choices.extend(transpose_choices)
        self.parser_sl_command.add_argument('--dataset', type=str.lower, 
            action="store", choices=dataset_choices, default="no-permutation", 
            help='Data file used for training. Default is using no permutation.')

    def _add_reinforcement_parser(self):
        # create the parser for the "reinforcement-learning" subcommand
        self.parser_rl_command = self.subparsers.add_parser('rl', 
            help='Reinforcement-learning help.')
        self.parser_rl_command.add_argument('--episodes', type=int, default=100, 
            help='Number of episodes to train in reinforcement-learning mode. Default is 100.')
        self.parser_rl_command.add_argument('--input_file', type=str.lower, default='./data/instances/taillard/ta41.txt',
                    help='The input_file represents the problem instance. Default is ./data/instances/taillard/ta41.txt')
        self.parser_rl_command.add_argument('--time_limit', type=int, default=60, 
            help='Time limit in seconds. Default is 60.')
        self.parser_rl_command.add_argument('--config_type', type=int, 
            action="store", choices=[1,2], default=1, 
            help='Hyperparameter config for training. Default is 1.')
        self.parser_rl_command.add_argument('--n_workers', type=int, default=4, 
            help='Amount of workers to run the experiments in parallel. Default is 4.')

    def _add_reinforcement_tuning_parser(self):
        # See also: https://wandb.ai/iamleonie/Intro-to-MLOps/reports/Intro-to-MLOps-Hyperparameter-Tuning--VmlldzozMTg2OTk3
        # create the parser for the "reinforcement-learning" subcommand
        self.parser_rl_tune_command = self.subparsers.add_parser('rl-tune', 
            help='Reinforcement-learning hyperparameter tuning help.')
        self.parser_rl_tune_command.add_argument('--tuning_method', type=str.lower, 
            action="store", choices=["bayes, grid, random"], default="bayes", 
            help='Tuning method for wandb sweeps. Default is bayes and cannot be parallelized.')
        self.parser_rl_tune_command.add_argument('--n_runs', type=int, default=60, 
            help='Amount of runs to perform for a parameter sweep. Default is 20.')
        self.parser_rl_tune_command.add_argument('--n_workers', type=int, default=1, 
            help='Amount of workers to run the experiments in parallel. Can only be used with grid and random tuning method. Default is 1.')
        self.parser_rl_tune_command.add_argument('--input_file', type=str.lower, default='./data/instances/taillard/ta41.txt',
                    help='The input_file represents the problem instance to perform hyperparameter tuning. Default is ./data/instances/taillard/ta41.txt')
        

    def _add_constraint_programming_parser(self):
        # create the parser for the "cp" subcommand (constraint programming)
        self.parser_cp_command = self.subparsers.add_parser('cp', 
            help='Constraint programming help.')
        self.parser_cp_command.add_argument('--input_files', metavar='FILE', nargs='+', 
            help='Input files to process')
        self.parser_cp_command.add_argument('--time_limit', type=int, default=0, 
            help='Time limit in minutes that the cp solver gets for solving a problem. Default is 0.')
        self.parser_cp_command.add_argument('--solution_type', type=str.lower, 
            action="store", choices=["feasible, optimal, all"], default="optimal", 
            help='Solution type that the cp solver should return Default it optimal.')


if __name__ == '__main__':
    logger = get_logger()
    args = CustomParser().args
    logger.info(f"Used arguments: {args}")

    if args.command == "cp":
        input_files = args.input_files
        time_limit = args.time_limit
        solution_type = args.solution_type

        for input_file in input_files:
            cp_solver = CPJobShopSolver(filename=input_file)
            cp_solver.solve(max_time=time_limit)

    if args.command == "rl":

        '''
        mean_makespan:  2618
        mean_reward:    103.0101010184735
        '''
        hyperparam_config_first = {
            "clip_range": 0.181648141774528,
            "ent_coef": 0.0033529692788612023,
            "gae_lambda": 0.9981645683766052,
            "gamma": 0.9278778323835192,
            "learning_rate": 0.001080234067815426,
            "max_grad_norm": 7.486785910278103,
            "n_epochs": 7,
            "n_steps": 731,
            "total_timesteps": 81947
        }
    
        '''
        mean_makespan: 2646
        mean_reward: 97.35353550687432
        '''
        hyperparam_config_second = {
            "clip_range": 0.2515491044924565,
            "ent_coef": 0.006207990430953167,
            "gae_lambda": 0.906079003617699,
            "gamma": 0.9041076240082796,
            "learning_rate": 0.002069479218298502,
            "max_grad_norm": 8.578211744760571,
            "n_epochs": 9,
            "n_steps": 1544,
            "total_timesteps": 69457
        }

        config_type = args.config_type
        time_limit_in_seconds = args.time_limit
        n_workers = args.n_workers
        episodes = args.episodes
        input_file = args.input_file

        config = None
        if config_type == 1:
            config = hyperparam_config_first
        else:
            config = hyperparam_config_second

        mean_reward, mean_makespan = train_agent_multi_env(hyperparam_config=config, n_envs=n_workers, input_file=input_file, time_limit_in_seconds=time_limit_in_seconds)
        print(f"Finished training with mean_reward of {mean_reward} and mean_makespan of {mean_makespan}.")

    if args.command == "rl-tune":
        
        method = args.tuning_method
        n_workers = args.n_workers
        n_runs = args.n_runs
        input_file = args.input_file

        run_sweep(tuning_method=method, n_runs=n_runs, n_workers=n_workers, input_file=input_file, project_name="maskable_ppo_hyperparameter_tuning")

    if args.command == "sl":
        pass

    if args.command == "dr":
        execute_fifo_worker(args.input_files)
        execute_mwkr_worker(args.input_files)
    
            

