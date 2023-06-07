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
from src.experiments.dispatching_rules_wandb import execute_fifo_worker, execute_mwkr_worker, execute_random_worker
from src.tuning.hyperparameter_search import run_sweep
from src.experiments.train_ppo_multi_env import train_agent_multi_env
from src.visualiztion.baselines import get_baseline_rl_makespans, get_baseline_rl_ta41_applied_to_others_makespans, evaluate_8_hour_ta41_rl, evaluate_rl_model
from src.supervised_learning.train_model import train


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
        self._add_visualization_parser()
        self._add_evaluation_parser()

        ######################################

        self.args = self.main_parser.parse_args()

    def _add_dispatching_rules_parser(self):
        # create the parser for the "supervised-learning" subcommand
        self.parser_dr_command = self.subparsers.add_parser('dr', 
            help='Dispatching rules help.')
        self.parser_dr_command.add_argument('--input_files', metavar='FILE', nargs='+', 
            help='Input files to process')
        self.parser_dr_command.add_argument('--dispatching_rule', type=str.lower, 
            action="store", choices=["fifo, mwkr, random, all"], default="all", 
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
        self.parser_rl_command.add_argument('--input_files', type=str.lower, metavar='FILE', nargs='+', 
            default='./data/instances/taillard/ta41.txt', help='Input files to process')
        self.parser_rl_command.add_argument('--time_limit', type=int, default=60, 
            help='Time limit in seconds. Default is 60.')
        self.parser_rl_command.add_argument('--config_type', type=int, 
            action="store", choices=[1,2,3,4], default=1, 
            help='Hyperparameter config for training. Default is 1.')
        self.parser_rl_command.add_argument('--n_workers', type=int, default=4, 
            help='Amount of workers to run the experiments in parallel. Default is 4.')

    def _add_reinforcement_tuning_parser(self):
        # See also: https://wandb.ai/iamleonie/Intro-to-MLOps/reports/Intro-to-MLOps-Hyperparameter-Tuning--VmlldzozMTg2OTk3
        # create the parser for the "reinforcement-learning" subcommand
        self.parser_rl_tune_command = self.subparsers.add_parser('rl-tune', 
            help='Reinforcement-learning hyperparameter tuning help.')
        self.parser_rl_tune_command.add_argument('--tuning_method', type=str.lower, 
            action="store", choices=["bayes", "grid", "random"], default="bayes", 
            help='Tuning method for wandb sweeps. Default is bayes and cannot be parallelized.')
        self.parser_rl_tune_command.add_argument('--n_runs', type=int, default=60, 
            help='Amount of runs to perform for a parameter sweep. Default is 20.')
        self.parser_rl_tune_command.add_argument('--max_episodes', type=int, default=30, 
            help='Amount of episodes per run to perform for a parameter sweep. Default is 30.')
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
        
    def _add_visualization_parser(self):
        # Add visualization functionality
        self.parser_visualize_command = self.subparsers.add_parser('visualize', 
            help='Visualization help.')
        self.parser_visualize_command.add_argument('--task', type=str.lower, 
            action="store", choices=["baseline_comparison, baseline_comparison_generalization"], default="baseline_comparison", 
            help='A task represents the visualization to execute.')
        self.parser_visualize_command.add_argument('--file_name', type=str.lower, default='plot.png',
                    help='Filename of the visualization/plot. Default is plot.png')
        self.parser_visualize_command.add_argument('--save_path', type=str.lower, default='./plots',
                    help='Path of the folder where to store --filename. Default is ./plots')
        
    def _add_evaluation_parser(self):
        # Add evaluation functionality
        self.parser_evaluate_command = self.subparsers.add_parser('evaluate', 
            help='Evaluation help.')
        self.parser_evaluate_command.add_argument('--model_type', type=str.lower, 
            action="store", choices=["rl", "supervised", "rl-sensitivity", "rl-generalization"], default="rl", 
            help='Type of model to evaluate: Reinforcement learning or supervised learning. Default is rl.')
        self.parser_evaluate_command.add_argument('--model_path', type=str.lower, default="./models/trained_tuned_30_mins/ta41/best_model.zip",
                    help='Path to model that you want to evaluate. Default is ./models/trained_tuned_30_mins/ta41/best_model.zip')
        self.parser_evaluate_command.add_argument('--instance_paths', metavar='FILE', nargs='+', default=["./data/instances/taillard/ta41.txt"],
            help='Paths to scheduling instances for evaluation. Default is ./data/instances/taillard/ta41.txt')
        
        transpose_choices = [f"transpose={x}" for x in range(0,16)]
        self.parser_evaluate_command.add_argument('--permutation_mode', type=str.lower, 
            action="store", choices=transpose_choices, default="transpose=0", 
            help='Data file used for training. Default is transpose=0, i.e., using no permutation.')
        
        


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

        
        hyperparam_config_tassel_architecture = {
            "batch_size": 140,
            "clip_range": 0.5403903472274568,
            "ent_coef": 0.0006468446031627639,
            "vf_coef": 0.5229849357873716,
            "gae_lambda": 0.9248978828624046,
            "gamma": 0.977158572561722,
            "learning_rate": 0.005108302278783693,
            "max_grad_norm": 5.481066781040825,
            "n_epochs": 14,
            "n_steps": 107
        }
        



        hyperparam_config_tassel = {
            "batch_size": 64,
            "clip_range": 0.541,
            "ent_coef": 0.496,
            "vf_coef": 0.7918,
            "gae_lambda": 0.9981645683766052,
            "gamma": 0.99,
            "learning_rate": 0.001080234067815426,
            "max_grad_norm": 7.486785910278103,
            "n_epochs": 12,
            "n_steps": 704
        }



        '''
        mean_makespan:  2618
        mean_reward:    103.0101010184735
        '''
        hyperparam_config_first = {
            "batch_size": 64,
            "clip_range": 0.181648141774528,
            "ent_coef": 0.0033529692788612023,
            "vf_coef": 0.5,
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
            "batch_size": 64,
            "clip_range": 0.2515491044924565,
            "ent_coef": 0.006207990430953167,
            "vf_coef": 0.5,
            "gae_lambda": 0.906079003617699,
            "gamma": 0.9041076240082796,
            "learning_rate": 0.002069479218298502,
            "max_grad_norm": 8.578211744760571,
            "n_epochs": 9,
            "n_steps": 1544,
            "total_timesteps": 69457
        }

        '''
        mean_makespan: 2676
        mean_reward: 97.35353550687432
        '''
        hyperparam_config_third = {
            "batch_size": 81,
            "clip_range": 0.4569412545379863,
            "ent_coef": 0.006403575874592836,
            "vf_coef": 0.541683487811123,
            "gae_lambda": 0.9971955082313728,
            "gamma": 0.9562334910680252,
            "learning_rate": 0.00004650540613946862,
            "max_grad_norm": 6.577071418772267,
            "n_epochs": 22,
            "n_steps": 990,
            "total_timesteps": 69457
        }


        config_type = args.config_type
        time_limit_in_seconds = args.time_limit
        n_workers = args.n_workers
        episodes = args.episodes

        input_files = args.input_files

        config = None
        if config_type == 1:
            config = hyperparam_config_first
        elif config_type == 2:
            config = hyperparam_config_second
        elif config_type == 3:
            config = hyperparam_config_tassel_architecture
        elif config_type == 4:
            config = hyperparam_config_third

        for input_file in input_files:
            print(f"Start training model for instance: {input_file}")
            mean_reward, mean_makespan = train_agent_multi_env(hyperparam_config=config, n_envs=n_workers, input_file=input_file, time_limit_in_seconds=time_limit_in_seconds)
            print(f"Finished training with mean_reward of {mean_reward} and mean_makespan of {mean_makespan}.")

    if args.command == "rl-tune":
        
        method = args.tuning_method
        n_workers = args.n_workers
        n_runs = args.n_runs
        input_file = args.input_file
        max_episodes = args.max_episodes

        run_sweep(tuning_method=method, n_runs=n_runs, n_workers=n_workers, max_episodes=max_episodes, input_file=input_file, project_name="maskable_ppo_hyperparameter_tuning")

    if args.command == "sl":

        dropouts = [False, True]

        data_sets = [
            {"ta42": [
                {   "name": "no_permutation",
                    "path": "30mins_tuned_policy/ta42/experiences_no-permutation_1000-episodes.npz"},
                {
                    "name": "20_percent_permutation", 
                    "path": "30mins_tuned_policy/ta42/experiences_transpose-3_1000-episodes.npz"},
                {
                    "name": "40_percent_permutation", 
                    "path": "30mins_tuned_policy/ta42/experiences_transpose-6_1000-episodes.npz"},
                {
                    "name": "60_percent_permutation",
                    "path": "30mins_tuned_policy/ta42/experiences_transpose-9_1000-episodes.npz"},
                {
                    "name": "80_percent_permutation", 
                    "path": "30mins_tuned_policy/ta42/experiences_transpose-12_1000-episodes.npz"},
                {
                    "name": "100_percent_permutation", 
                    "path": "30mins_tuned_policy/ta42/experiences_transpose-15_1000-episodes.npz"},
            ]},
            {"ta43": [
                {   "name": "no_permutation",
                    "path": "30mins_tuned_policy/ta43/experiences_no-permutation_1000-episodes.npz"},
                {
                    "name": "20_percent_permutation", 
                    "path": "30mins_tuned_policy/ta43/experiences_transpose-3_1000-episodes.npz"},
                {
                    "name": "40_percent_permutation", 
                    "path": "30mins_tuned_policy/ta43/experiences_transpose-6_1000-episodes.npz"},
                {
                    "name": "60_percent_permutation",
                    "path": "30mins_tuned_policy/ta43/experiences_transpose-9_1000-episodes.npz"},
                {
                    "name": "80_percent_permutation", 
                    "path": "30mins_tuned_policy/ta43/experiences_transpose-12_1000-episodes.npz"},
                {
                    "name": "100_percent_permutation", 
                    "path": "30mins_tuned_policy/ta43/experiences_transpose-15_1000-episodes.npz"},
            ]},
            {"ta44": [
                {   "name": "no_permutation",
                    "path": "30mins_tuned_policy/ta44/experiences_no-permutation_1000-episodes.npz"},
                {
                    "name": "20_percent_permutation", 
                    "path": "30mins_tuned_policy/ta44/experiences_transpose-3_1000-episodes.npz"},
                {
                    "name": "40_percent_permutation", 
                    "path": "30mins_tuned_policy/ta44/experiences_transpose-6_1000-episodes.npz"},
                {
                    "name": "60_percent_permutation",
                    "path": "30mins_tuned_policy/ta44/experiences_transpose-9_1000-episodes.npz"},
                {
                    "name": "80_percent_permutation", 
                    "path": "30mins_tuned_policy/ta44/experiences_transpose-12_1000-episodes.npz"},
                {
                    "name": "100_percent_permutation", 
                    "path": "30mins_tuned_policy/ta44/experiences_transpose-15_1000-episodes.npz"},
            ]},
            {"ta45": [
                {   "name": "no_permutation",
                    "path": "30mins_tuned_policy/ta45/experiences_no-permutation_1000-episodes.npz"},
                {
                    "name": "20_percent_permutation", 
                    "path": "30mins_tuned_policy/ta45/experiences_transpose-3_1000-episodes.npz"},
                {
                    "name": "40_percent_permutation", 
                    "path": "30mins_tuned_policy/ta45/experiences_transpose-6_1000-episodes.npz"},
                {
                    "name": "60_percent_permutation",
                    "path": "30mins_tuned_policy/ta45/experiences_transpose-9_1000-episodes.npz"},
                {
                    "name": "80_percent_permutation", 
                    "path": "30mins_tuned_policy/ta45/experiences_transpose-12_1000-episodes.npz"},
                {
                    "name": "100_percent_permutation", 
                    "path": "30mins_tuned_policy/ta45/experiences_transpose-15_1000-episodes.npz"},
            ]},
            {"ta46": [
                {   "name": "no_permutation",
                    "path": "30mins_tuned_policy/ta46/experiences_no-permutation_1000-episodes.npz"},
                {
                    "name": "20_percent_permutation", 
                    "path": "30mins_tuned_policy/ta46/experiences_transpose-3_1000-episodes.npz"},
                {
                    "name": "40_percent_permutation", 
                    "path": "30mins_tuned_policy/ta46/experiences_transpose-6_1000-episodes.npz"},
                {
                    "name": "60_percent_permutation",
                    "path": "30mins_tuned_policy/ta46/experiences_transpose-9_1000-episodes.npz"},
                {
                    "name": "80_percent_permutation", 
                    "path": "30mins_tuned_policy/ta46/experiences_transpose-12_1000-episodes.npz"},
                {
                    "name": "100_percent_permutation", 
                    "path": "30mins_tuned_policy/ta46/experiences_transpose-15_1000-episodes.npz"},
            ]}
        ]


        
        models = [
            {"ta41": "models/trained_tuned_30_mins/ta41/best_model.zip"},
            {"ta42": "models/trained_tuned_30_mins/ta42/best_model.zip"},
            {"ta43": "models/trained_tuned_30_mins/ta43/best_model.zip"},
            {"ta44": "models/trained_tuned_30_mins/ta44/best_model.zip"},
            {"ta45": "models/trained_tuned_30_mins/ta45/best_model.zip"},
            {"ta46": "models/trained_tuned_30_mins/ta46/best_model.zip"},
            {"ta47": "models/trained_tuned_30_mins/ta47/best_model.zip"},
            {"ta48": "models/trained_tuned_30_mins/ta48/best_model.zip"},
            {"ta49": "models/trained_tuned_30_mins/ta49/best_model.zip"},
            {"ta50": "models/trained_tuned_30_mins/ta50/best_model.zip"},
        ]

        '''

        '''
                  
        '''
        1. Go through every data set collected by ta41 policy
            1a) Train architecture without dropout
            1b) Train architecture with dropout
        
        -[ ] Set time limit to 30 mins
        -[ ] Set proper save path for best performing model
        -[ ] Track metrics using w&bs
        -[ ] Upload best performing model to w&b
        -[ ] Evaluate models on all datasets
        '''


        for data_set in data_sets:
            # data_set is dict
            for data_set_name, permutation_set in data_set.items():
                for permutation_dict in permutation_set:
                    for dropout in dropouts:
                        permutation_name = permutation_dict['name']
                        permutation_path = permutation_dict['path']

                        print(f"Using data set {data_set_name} with {permutation_name} loaded from {permutation_path}")
                        print(f"Dropout enabled: {dropout}")
                        train(data_filename = permutation_path, instance_name = data_set_name, data_desc = permutation_name,
                              use_dropout = dropout, lr = 0.001, num_classes = 30+1, input_size = 30*7, num_epochs = 1000000, time_limit = 60*30
                              )




    if args.command == "dr":
        execute_fifo_worker(args.input_files)
        execute_mwkr_worker(args.input_files)
        #execute_random_worker(args.input_files, n_runs=100)

    if args.command == "visualize":
        task_name = args.task
        file_name = args.file_name
        save_path = args.save_path

        print(f"Task name: {task_name}")
        print(f"File name: {file_name}")
        print(f"Save path: {save_path}")

        evaluate_8_hour_ta41_rl()
        #get_baseline_rl_makespans()
        #get_baseline_rl_ta41_applied_to_others_makespans()

    if args.command == "evaluate":



        '''
                self.parser_evaluate_command.add_argument('--model_type', type=str.lower, 
            action="store", choices=["rl, supervised"], default="rl", 
            help='Type of model to evaluate: Reinforcement learning or supervised learning. Default is rl.')
        self.parser_evaluate_command.add_argument('--model_path', type=str.lower, default="./models/trained_tuned_30_mins/ta41/best_model.zip",
                    help='Path to model that you want to evaluate. Default is ./models/trained_tuned_30_mins/ta41/best_model.zip')
        self.parser_evaluate_command.add_argument('--instance_paths', metavar='FILE', nargs='+', default=["./data/instances/taillard/ta41.txt"],
            help='Paths to scheduling instances for evaluation. Default is ./data/instances/taillard/ta41.txt')
        '''

        model_type = args.model_type # --model_type, options=["rl, supervised"]
        model_path = args.model_path # --model_path, default="./models/trained_tuned_30_mins/ta41/best_model.zip"
        instance_paths = args.instance_paths # --instance_paths, default=["./data/instances/taillard/ta41.txt"]
        permutation_mode = args.permutation_mode

        if model_type == "rl-generalization":
            policy_names = ["ta41", "ta42", "ta43", "ta44", "ta45", "ta46", "ta47", "ta48", "ta49", "ta50"]
            instance_names = ["ta41", "ta42", "ta43", "ta44", "ta45", "ta46", "ta47", "ta48", "ta49", "ta50"]

            makespans = []

            # Iterate over each policy
            for policy_name in policy_names:
                print(f"Calculate makespans for policy {policy_name}")

                policy_makespans = []

                model_path = f"./models/trained_tuned_30_mins/{policy_name}/best_model.zip"

                # Apply policy to each instance
                for instance_name in instance_names:
                    print(f"Instance: {instance_name}")
                    instance_path = f"./data/instances/taillard/{instance_name}.txt"
                
                    mean_reward, mean_makespan, elapsed_time = evaluate_rl_model(model_path=model_path, eval_instance_path=instance_path, eval_permutation_mode=None, n_eval_episodes=1)
                    policy_makespans.append({instance_name: mean_makespan})

                makespans.append({policy_name: policy_makespans})
            
            print(makespans)

    

        if model_type == "rl-sensitivity":
            permutation_modes = ["transpose=0", "transpose=3", "transpose=6", "transpose=9", "transpose=12", "transpose=15"]
            instance_names = ["ta41", "ta42", "ta43", "ta44", "ta45", "ta46", "ta47", "ta48", "ta49", "ta50"]
            n_eval_episodes = 100

            makespans = []

            for instance_name in instance_names:
                print(f"Calculate makespans for instance {instance_name}...")

                instance_makespans = []

                instance_path = f"./data/instances/taillard/{instance_name}.txt"
                model_path = f"./models/trained_tuned_30_mins/{instance_name}/best_model.zip"

                for permutation_mode in permutation_modes:
                    print(f"Permutation mode: {permutation_mode}")

                    if permutation_mode == "transpose=0":
                        permutation_mode = None
                
                    mean_reward, mean_makespan, elapsed_time = evaluate_rl_model(model_path=model_path, eval_instance_path=instance_path, eval_permutation_mode=permutation_mode, n_eval_episodes=100)
                    instance_makespans.append({permutation_mode: mean_makespan})

                makespans.append({instance_name: instance_makespans})
            
            print(makespans)






        if model_type == "rl":
            print(f"Model type: {model_type}")
            print(f"Model path: {model_path}")
            print(f"Instance paths: {instance_paths}")
            print(f"Permutation mode: {permutation_mode}")

            if permutation_mode == "transpose=0":
                permutation_mode = None

            for instance_path in instance_paths:
                evaluate_rl_model(model_path=model_path, eval_instance_path=instance_path, eval_permutation_mode=permutation_mode)
            
        elif model_type == "supervised":
            pass



    
            

