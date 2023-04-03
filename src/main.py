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

def parse_arguments():
    '''
    Different experiments:
        1. Supervised learning (train, eval, confusion_matrix, loss, acc, auc)
            c) plot_path
            d) model_path
            e) dataset: no-permutation, random, transpose-{1:15}
            h) hidden_size
    '''


    # the hyphen makes the argument optional
    '''
    parser.add_argument('-gpu', type=str, default='0', help='GPU: 0 or 1. Default is 0.')
    parser.add_argument('-episodes', type=int, default=150, help='Number of games/episodes to play. Default is 150.')
    parser.add_argument('-alpha', type=float, default=0.0001, help='Learning rate alpha for the actor network. Default is 0.0001.')
    parser.add_argument('-beta', type=float, default=0.001, help='Learning rate beta for the critic network. Default is 0.001.')
    parser.add_argument('-gamma', type=float, default=0.99, help='Discount factor for update equation')
    parser.add_argument('-tau', type=float, default=0.001, help='Update network parameters. Default is 0.001.')
    parser.add_argument('-algo', type=str, default='DDPGAgent',
                    help='You can use the following algorithms: DDPGAgent. Default is DDPGAgent.')
    parser.add_argument('-buffer_size', type=int, default=1000000, help='Maximum size of memory/replay buffer. Default is 1000000.')
    parser.add_argument('-batch_size', type=int, default=128, help='Batch size for training. Default is 128.')
    parser.add_argument('-load_checkpoint', type=bool, default=False,
                        help='Load model checkpoint/weights. Default is False.')
    parser.add_argument('-model_path', type=str, default='data/',
                        help='Path for model saving/loading. Default is data/')
    parser.add_argument('-plot_path', type=str, default='plots/',
                        help='Path for saving plots. Default is plots/')
    parser.add_argument('-save_plot', type=bool, default=True,
                        help='Save plot of eval or/and training phase. Default is True.')

    parser.add_argument('-multiagent_env', type=bool, default=False,
                        help='Using the multi agent environment version. Default is False.')
    parser.add_argument('-visual_env', type=bool, default=False,
                        help='Using the visual environment. Default is False.')
    '''
    
    #args = parser.parse_args()
    #return args

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
        pass

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
    
            
