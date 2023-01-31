import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
from environments.DMP_Env_2D_dynamic import deep_mobile_printing_2d1r
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plan-choice', default=0, type=int,
                    help='0 dense, 1 sparse')
parser.add_argument('--seed', default=1, type=int,
                    help='random seed from 1 through 4')
args = parser.parse_args()

PLAN_LIST=["dense","sparse"]
PLAN_NAME=PLAN_LIST[args.plan_choice]
config = Config()
config.seed = args.seed
config.environment = deep_mobile_printing_2d1r(data_path="../../../Env/2D/data_2d_dynamic_"+PLAN_NAME+"_envplan_500_train.pkl")
config.environment_val = deep_mobile_printing_2d1r(data_path="../../../Env/2D/data_2d_dynamic_"+PLAN_NAME+"_envplan_500_val.pkl",random_choose_paln=False) 
config.num_episodes_to_run = 10000
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.GPU = "cuda:0"
config.overwrite_existing_results_file = True
config.randomise_random_seed = False
config.save_model = False
OUT_FILE_NAME="SAC_2d_"+PLAN_NAME+"_seed_"+str(config.seed)
config.save_model_path = "./SAC/2D/dynamic/"+OUT_FILE_NAME+"/"
config.file_to_save_data_results = "./SAC/2D/dynamic/"+OUT_FILE_NAME+"/"+"Results_Data.pkl"
config.file_to_save_results_graph = "./SAC/2D/dynamic/"+OUT_FILE_NAME+"/"+"Results_Graph.png"
if os.path.exists(config.save_model_path) == False:
    os.makedirs(config.save_model_path)

config.hyperparameters = {

    "Actor_Critic_Agents":  {

        "learning_rate": 0.005,
        "linear_hidden_units": [512, 512],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 5.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 1.0,
        "normalise_rewards": False,
        "exploration_worker_difference": 2.0,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [512, 512, 512],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "Critic": {
            "learning_rate": 0.0003,
            "linear_hidden_units": [512, 512,512],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.005,
            "gradient_clipping_norm": 5,
            "initialiser": "Xavier"
        },

        "min_steps_before_learning": 400,
        "batch_size": 100,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 3,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True
    }
}

if __name__ == "__main__":

    # turn it on if you want to test specific network saved

    test = False
    dictPath = None
    AGENTS = [SAC_Discrete]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents(test, dictPath)