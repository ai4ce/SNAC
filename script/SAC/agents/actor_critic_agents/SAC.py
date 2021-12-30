from agents.Base_Agent import Base_Agent
from utilities.OU_Noise import OU_Noise
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import time
from environments.DMP_Env_1D_dynamic_test import deep_mobile_printing_1d1r
# from environments.DMP_Env_2D_static import deep_mobile_printing_2d1r
from environments.DMP_Env_2D_dynamic_test import deep_mobile_printing_2d1r
# from environments.DMP_simulator_3d_static_circle_test import deep_mobile_printing_3d1r
# from environments.DMP_simulator_3d_dynamic_triangle_test import deep_mobile_printing_3d1r

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6



class SAC(Base_Agent):
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "CONTINUOUS", "Action types must be continuous. Use SAC Discrete instead for discrete actions"
        assert self.config.hyperparameters["Actor"]["final_layer_activation"] != "Softmax", "Final actor layer must not be softmax"
        self.hyperparameters = config.hyperparameters
        self.critic_local = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1, key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic", override_seed=self.config.seed + 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                           key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size + self.action_size, output_dim=1,
                                            key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, device=self.device)
        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size * 2, key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.environment.action_space.shape).to(self.device)).item() # heuristic value from the paper
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]

        self.add_extra_noise = self.hyperparameters["add_extra_noise"]
        if self.add_extra_noise:
            self.noise = OU_Noise(self.action_size, self.config.seed, self.hyperparameters["mu"],
                                  self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

    def save_result(self):
        """Saves the result of an episode of the game. Overriding the method in Base Agent that does this because we only
        want to keep track of the results during the evaluation episodes"""
        if self.episode_number == 1 or not self.do_evaluation_iterations:
            self.game_full_episode_scores.extend([self.total_episode_score_so_far])
            self.rolling_results.append(np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]))
            self.save_max_result_seen()

        elif (self.episode_number - 1) % TRAINING_EPISODES_PER_EVAL_EPISODE == 0:
            self.game_full_episode_scores.extend([self.total_episode_score_so_far for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.rolling_results.extend([np.mean(self.game_full_episode_scores[-1 * self.rolling_score_window:]) for _ in range(TRAINING_EPISODES_PER_EVAL_EPISODE)])
            self.save_max_result_seen()

    def reset_game(self):
        """Resets the game information so we are ready to play a new episode"""
        Base_Agent.reset_game(self)
        if self.add_extra_noise: self.noise.reset()

    def step(self, isEval=False):
        if not isEval:
            """Runs an episode on the game, saving the experience and running a learning step if appropriate"""
            eval_ep = self.episode_number % TRAINING_EPISODES_PER_EVAL_EPISODE == 0 and self.do_evaluation_iterations
            self.episode_step_number_val = 0
            while not self.done:
                self.episode_step_number_val += 1
                self.action = self.pick_action(eval_ep)
                
                self.conduct_action(self.action)
                if self.time_for_critic_and_actor_to_learn():
                    for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                        self.learn()
                mask = False if self.episode_step_number_val >= 1000. else self.done
                if not eval_ep: self.save_experience(experience=(self.state, self.action, self.reward, self.next_state, mask))
                self.state = self.next_state
                self.global_step_number += 1
            print(self.total_episode_score_so_far)
            if eval_ep: self.print_summary_of_latest_evaluation_episode()
            if eval_ep:
                iou_test = self.environment.iou()
                # iou_test=self.iou(self.environment.environment_memory,np.argmax(self.environment.one_hot)+1)
                print('\nEpodise: ', self.episode_number,'| Ep_reward_test:', self.reward)
                print('\nEpodise: ', self.episode_number,'| Ep_IOU_test: ', iou_test)
                self.reward_history_test.append(self.reward)
                self.iou_history_test.append(iou_test)
            self.episode_number += 1
        else:
            eval_ep=True

            '''
            ########################3d dynamic test################################          
            print("Testing")

            plan_choose = self.environment.plan_choose
            print(f"PLAN = {plan_choose}")
            iou_all_average = 0
            iou_all_min = 1
            for test_set in range(10):
                env = deep_mobile_printing_3d1r(plan_choose=plan_choose, test_set=test_set)


                def iou(environment_memory,environment_plan,HALF_WINDOW_SIZE,plan_height,plan_width):
                    component1=environment_plan[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
                                    HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
                    component2=environment_memory[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
                                    HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
                    overlap = component1*component2 # Logical AND
                    union = component1 + component2 # Logical OR
                    IOU = overlap.sum()/float(union.sum())
                    return IOU

                print(test_set)
                N_iteration_test = 200
                best_iou = 0
                iou_test_total = 0
                iou_min = 1
                reward_test_total = 0
                start_time_test = time.time()

                fig = plt.figure(figsize=[10, 5])
                ax1 = fig.add_subplot(1, 2, 1, projection='3d')
                ax2 = fig.add_subplot(1, 2, 2)

                for ep in range(N_iteration_test):
                    obs = env.reset()
                    reward_test = 0
                    self.state = obs
                    while True:
                        self.action = self.pick_action(eval_ep)
                        self.conduct_action(self.action)
                        obs, r, done = env.step(self.action)
                        self.state = obs
                        reward_test += r
                        if done:
                            break

                    iou_test = iou(env.environment_memory,env.plan,env.HALF_WINDOW_SIZE,env.plan_height,env.plan_width)
                    iou_min = min(iou_min, iou_test)

                    if iou_test > best_iou:
                        best_iou = iou_test
                        best_plan = env.plan
                        best_step = env.count_step
                        best_brick = env.count_brick
                        best_tb = env.total_brick
                        best_env = env.environment_memory
                        env.render(ax1, ax2)
                        save_path = "plots/"
                        plt.savefig(save_path+"SAC_Plan"+str(test_set)+'_'+str(self.environment.plan_choose)+'_good.png')

                    iou_test_total += iou_test
                    reward_test_total += reward_test

                reward_test_total = reward_test_total / N_iteration_test
                iou_test_total = iou_test_total / N_iteration_test
                secs = int(time.time() - start_time_test)
                mins = secs // 60
                secs = secs % 60
                print(f"time = {mins} min {secs} sec")
                print(f"iou = {iou_test_total}")
                print(f"reward_test = {reward_test_total}")
                if best_iou>0:
                    env.render(ax1,ax2,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,best_iou=best_iou,best_step=best_step,best_brick=best_brick)
                else:
                    env.render(ax1,ax2,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test)
                save_path = "plots/"
                plt.savefig(save_path+"SAC_Plan"+str(test_set)+'_'+str(self.environment.plan_choose)+'_summary.png')

                iou_all_average += iou_test_total
                iou_all_min = min(iou_min,iou_all_min)

            iou_all_average = iou_all_average/10
            print('iou_all_average',iou_all_average)
            print('iou_all_min',iou_all_min)
            '''

            
































            
            '''
            ########################3d static test################################
            print(f"Testing plan {self.environment.plan_choose}")

            env = deep_mobile_printing_3d1r(plan_choose=self.environment.plan_choose)


            def iou(environment_memory,environment_plan,HALF_WINDOW_SIZE,plan_height,plan_width):
                component1=environment_plan[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
                                HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
                component2=environment_memory[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
                                HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
                overlap = component1*component2 # Logical AND
                union = component1 + component2 # Logical OR
                IOU = overlap.sum()/float(union.sum())
                return IOU

            N_iteration_test = 500 
            best_iou = 0
            iou_test_total = 0
            iou_min = 1
            reward_test_total = 0
            start_time_test = time.time()

            fig = plt.figure(figsize=[10, 5])
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2)

            for ep in range(N_iteration_test):
                obs = env.reset()
                reward_test = 0
                self.state = obs
                
                while True:

                    self.action = self.pick_action(eval_ep)
                    self.conduct_action(self.action)
                    # action, _ = test_agent.predict(obs)
                    obs, r, done = env.step(self.action)
                    self.state = obs

                    # action, _ = test_agent.predict(obs)
                    # obs, r, done, info = env.step(action)
                    reward_test += r
                    if done:
                        break

                iou_test = iou(env.environment_memory,env.plan,env.HALF_WINDOW_SIZE,env.plan_height,env.plan_width)
                iou_min = min(iou_min, iou_test)

                if iou_test > best_iou:
                    best_iou = iou_test
                    best_plan = env.plan
                    best_step = env.count_step
                    best_brick = env.count_brick
                    best_tb = env.total_brick
                    best_env = env.environment_memory
                    env.render(ax1, ax2)
                    save_path = "plots/"
                    plt.savefig(save_path+"SAC_Plan"+'_'+str(self.environment.plan_choose)+'_good.png')
                iou_test_total += iou_test
                reward_test_total += reward_test

            reward_test_total = reward_test_total / N_iteration_test
            iou_test_total = iou_test_total / N_iteration_test
            secs = int(time.time() - start_time_test)
            mins = secs // 60
            secs = secs % 60
            print(f"time = {mins} min {secs} sec")
            print(f"iou = {iou_test_total}")
            print(f"reward_test = {reward_test_total}")

            if best_iou>0:
                env.render(ax1,ax2,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,best_iou=best_iou,best_step=best_step,best_brick=best_brick)
            else:
                env.render(ax1,ax2,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test)
            save_path = "plots/"
            plt.savefig(save_path+"SAC_Plan"+'_'+str(self.environment.plan_choose)+'_summary.png')
            '''
            

















































            
            # ########################2d dynamic test################################
            # print("Testing")

            
            # print(f"PLAN = {self.environment.plan_choose}")
            # iou_all_average = 0
            # iou_all_min = 1
            # for test_set in range(10):
            #     test_set = 6
            #     env = deep_mobile_printing_2d1r(plan_choose=self.environment.plan_choose, test_set=test_set)

            #     def iou(environment_memory,environment_plan,HALF_WINDOW_SIZE,plan_height,plan_width):
            #         component1=environment_plan[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
            #                             HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
            #         component2=environment_memory[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
            #                             HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
            #         overlap = component1*component2 # Logical AND
            #         union = component1 + component2 # Logical OR
            #         IOU = overlap.sum()/float(union.sum())
            #         return IOU

            #     print(test_set)
            #     N_iteration_test = 200
            #     best_iou = 0
            #     iou_test_total = 0
            #     iou_min = 1
            #     reward_test_total = 0
            #     start_time_test = time.time()

            #     fig = plt.figure(figsize=(5, 5))
            #     ax = fig.add_subplot(1, 1, 1)
            #     for ep in range(N_iteration_test):
            #         obs = env.reset()
            #         reward_test = 0
            #         self.state = obs
            #         while True:
            #             self.action = self.pick_action(eval_ep)
            #             self.conduct_action(self.action)
            #             # action, _ = test_agent.predict(obs)
            #             obs, r, done, info = env.step(self.action)
            #             self.state = obs
            #             reward_test += r
            #             if done:
            #                 break

            #         iou_test = iou(env.environment_memory,env.plan,env.HALF_WINDOW_SIZE,env.plan_height,env.plan_width)
            #         iou_min = min(iou_min, iou_test)

            #         if iou_test > best_iou:
            #             best_iou = iou_test
            #             best_plan = env.plan
            #             best_tb = env.total_brick
            #             env.render(ax)
            #             save_path = "plots/"
            #             plt.savefig(save_path+"SAC_Plan"+str(test_set)+'_'+str(self.environment.plan_choose)+'_good.png')
            #         iou_test_total += iou_test
            #         reward_test_total += reward_test

            #     reward_test_total = reward_test_total / N_iteration_test
            #     iou_test_total = iou_test_total / N_iteration_test
            #     secs = int(time.time() - start_time_test)
            #     mins = secs // 60
            #     secs = secs % 60
            #     print(f"time = {mins} min {secs} sec")
            #     print(f"iou = {iou_test_total}")
            #     print(f"reward_test = {reward_test_total}")
            #     env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test)
            #     iou_all_average += iou_test_total
            #     iou_all_min = min(iou_min,iou_all_min)
            #     save_path = "plots/"
            #     plt.savefig(save_path+"SAC_Plan"+str(test_set)+'_'+str(self.environment.plan_choose)+'_summary.png')

            # iou_all_average = iou_all_average/10
            # print('iou_all_average',iou_all_average)
            # print('iou_all_min',iou_all_min)














            
            '''
            ########################2d static test################################
            def iou(environment_memory,environment_plan,HALF_WINDOW_SIZE,plan_height,plan_width):
                component1=environment_plan[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
                                HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
                component2=environment_memory[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
                                HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
                overlap = component1*component2 # Logical AND
                union = component1 + component2 # Logical OR
                IOU = overlap.sum()/float(union.sum())
                return IOU

            env = deep_mobile_printing_2d1r(plan_choose=self.environment.plan_choose)

            N_iteration_test = 500
            best_iou = 0
            iou_test_total = 0
            iou_min = 1
            reward_test_total = 0
            start_time_test = time.time()

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)
            for ep in range(N_iteration_test):
                obs = env.reset()
                reward_test = 0
                self.state = obs
                while True:
                    self.action = self.pick_action(eval_ep)
                    self.conduct_action(self.action)
                    obs, r, done = env.step(self.action)
                    self.state = obs

                    # action, _ = test_agent.predict(obs)
                    # obs, r, done, info = env.step(action)
                    reward_test += r
                    if done:
                        break

                iou_test = iou(env.environment_memory,env.plan,env.HALF_WINDOW_SIZE,env.plan_height,env.plan_width)
                iou_min = min(iou_min, iou_test)

                if iou_test > best_iou:
                    best_iou = iou_test
                    best_plan = env.plan
                    best_tb = env.total_brick
                    env.render(ax)
                    save_path = "plots/"
                    plt.savefig(save_path+"SAC_Plan"+str(self.environment.plan_choose)+'_good.png')
                iou_test_total += iou_test
                reward_test_total += reward_test

            reward_test_total = reward_test_total / N_iteration_test
            iou_test_total = iou_test_total / N_iteration_test
            secs = int(time.time() - start_time_test)
            mins = secs // 60
            secs = secs % 60
            print(f"time = {mins} min {secs} sec")
            print(f"iou = {iou_test_total}")
            print(f"reward_test = {reward_test_total}")
            env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test)

            save_path = "plots/"
            plt.savefig(save_path+"SAC_Plan"+str(self.environment.plan_choose)+'_summary.png')
            '''
            





















            

            '''
            ########################1d dynamic test################################

            iou_all_average = 0
            iou_all_min = 1
            for plan_choose in range(10):
                plan_choose = 8
                print(plan_choose)
                env = deep_mobile_printing_1d1r(plan_choose=plan_choose)
                N_iteration_test = 200
                best_iou = 0
                iou_test_total = 0
                iou_min = 1
                reward_test_total = 0
                start_time_test = time.time()

                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(1, 1, 1)
                for ep in range(N_iteration_test):
                    obs = env.reset()
                    reward_test = 0
                    self.state = obs
                    while True:
                        self.action = self.pick_action(eval_ep)
                        self.conduct_action(self.action)
                        obs, r, done  = env.step(self.action)
                        self.state = obs
                        reward_test += r
                        if done:
                            break

                    iou_test = env.iou()
                    iou_min = min(iou_min, iou_test)

                    if iou_test > best_iou:
                        best_iou = iou_test
                        best_plan = env.plan
                        best_tb = env.total_brick
                        env.render(ax)
                        save_path = "plots/"
                        plt.savefig(save_path+"SAC_Plan"+str(plan_choose)+'_good.png')
                    iou_test_total += iou_test
                    reward_test_total += reward_test

                reward_test_total = reward_test_total / N_iteration_test
                iou_test_total = iou_test_total / N_iteration_test
                secs = int(time.time() - start_time_test)
                mins = secs // 60
                secs = secs % 60
                print(f"time = {mins} min {secs} sec")
                print(f"iou = {iou_test_total}")
                print(f"reward_test = {reward_test_total}")
                env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test)
                iou_all_average += iou_test_total
                iou_all_min = min(iou_min,iou_all_min)
                save_path = "plots/"
                plt.savefig(save_path+"SAC_Plan"+str(plan_choose)+'_summary.png')

            iou_all_average = iou_all_average/10
            print('iou_all_average',iou_all_average)
            print('iou_all_min',iou_all_min)
            '''
























            
        
            ########################1d static test################################
            plan = self.environment.plan_choose

            print(f"Testing plan {plan}")

            N_iteration_test = 1
            best_iou = 0
            iou_test_total = 0
            iou_min = 1
            reward_test_total = 0
            start_time_test = time.time()

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)
            for ep in range(N_iteration_test):
                obs = self.environment.reset()
                reward_test = 0

                while True:
                    self.action = self.pick_action(eval_ep)
                    self.conduct_action(self.action)
                    self.state = self.next_state
                    obs, r, done = self.environment.step(self.action)
                    reward_test += r
                    if done:
                        break
                # self.environment.render(ax)
                # plt.show()
                iou_test = self.environment.iou()
                iou_min = min(iou_min, iou_test)

                if iou_test > best_iou:
                    best_iou = iou_test
                    best_plan = self.environment.plan
                    best_tb = self.environment.total_brick
                    self.environment.render(ax)
                    save_path = self.config.save_model_path
                    plt.savefig(save_path+"SAC_Plan"+str(plan)+'_good.png')
                iou_test_total += iou_test
                reward_test_total += reward_test

            reward_test_total = reward_test_total / N_iteration_test
            iou_test_total = iou_test_total / N_iteration_test
            secs = int(time.time() - start_time_test)
            mins = secs // 60
            secs = secs % 60
            print(f"time = {mins} min {secs} sec")
            print(f"iou = {iou_test_total}")
            print(f"reward_test = {reward_test_total}")
    
            self.environment.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test)

            save_path = self.config.save_model_path
            plt.savefig(save_path+"SAC_Plan"+str(plan)+'_summary.png')
        

















            
            ########################Initial test################################
            # fig = plt.figure(figsize=(5, 5))

            # ax = fig.add_subplot(1, 1, 1)

            # print(self.environment.total_brick)

            # print(self.environment.one_hot)

            # step = self.environment.total_step

            # ax.clear()
            # while not self.done:

            #     self.action = self.pick_action(eval_ep)
                
            #     self.conduct_action(self.action)

            #     self.state = self.next_state
                
            #     self.environment.render(ax)

            #     self.environment.iou()

            #     plt.pause(1e-6)

            # plt.show()
            exit()

    def pick_action(self, eval_ep, state=None):
        """Picks an action using one of three methods: 1) Randomly if we haven't passed a certain number of steps,
         2) Using the actor in evaluation mode if eval_ep is True  3) Using the actor in training mode if eval_ep is False.
         The difference between evaluation and training mode is that training mode does more exploration"""
        if state is None: state = self.state
        # print(self.state)
    
        if eval_ep: action = self.actor_pick_action(state=state, eval=True)
        elif self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            # print('pick_action')
            action = np.random.randint(0, self.environment.action_dim) 
            #action = self.environment.action_space.sample()
            print("Picking random action ", action)
        else:
            action = self.actor_pick_action(state=state)
        if self.add_extra_noise:
            action += self.noise.sample()
        return action

    def actor_pick_action(self, state=None, eval=False):
        """Uses actor to pick an action in one of two ways: 1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled 2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling"""
        if state is None: state = self.state
        state = torch.FloatTensor([state]).to(self.device)
        if len(state.shape) == 1: state = state.unsqueeze(0)
        if eval == False: action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        actor_output = self.actor_local(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  #rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def time_for_critic_and_actor_to_learn(self):
        """Returns boolean indicating whether there are enough experiences to learn from and it is time to learn for the
        actor and critic"""
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
               self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters["update_every_n_steps"] == 0

    def learn(self):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        if self.automatic_entropy_tuning: alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else: alpha_loss = None
        self.update_all_parameters(qf1_loss, qf2_loss, policy_loss, alpha_loss)

    def sample_experiences(self):
        return  self.memory.sample()

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(torch.cat((next_state_batch, next_state_action), 1))
            qf2_next_target = self.critic_target_2(torch.cat((next_state_batch, next_state_action), 1))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (min_qf_next_target)
        qf1 = self.critic_local(torch.cat((state_batch, action_batch), 1))
        qf2 = self.critic_local_2(torch.cat((state_batch, action_batch), 1))
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(torch.cat((state_batch, action), 1))
        qf2_pi = self.critic_local_2(torch.cat((state_batch, action), 1))
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_all_parameters(self, critic_loss_1, critic_loss_2, actor_loss, alpha_loss):
        """Updates the parameters for the actor, both critics and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def print_summary_of_latest_evaluation_episode(self):
        """Prints a summary of the latest episode"""
        print(" ")
        print("----------------------------")
        print("Episode score {} ".format(self.total_episode_score_so_far))
        print("----------------------------")