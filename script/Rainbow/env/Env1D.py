import math
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.utils import save_plot


class Env1DStatic(gym.Env):
    def __init__(self, args):
        self.plan_width = 30
        self.plan_height = 20

        self.plan_choose = (args.plan_choose if args.plan_choose is not None else 0)
        self.environment_height = 50
        self.environment_memory = None
        self.count_brick = None
        self.brick_memory = None
        self.HALF_WINDOW_SIZE = args.half_window_size
        self.environment_width = self.plan_width + 2 * self.HALF_WINDOW_SIZE
        self.wall = np.ones((1, 2)) * (-1)
        self.position_memory = None
        self.observation = None
        self.count_step = 0
        self.total_step = 750
        self.plan, self.total_brick = self.create_plan()
        self.action_dim = 3
        self.state_dim = args.half_window_size * 2 + 3
        self.window = args.history_length
        self.uniform_step = args.uniform_step
        self.step_size = 1

        self.state_buffer = deque([], maxlen=self.window)
        self.features = 7 - 1

    def set_plan_choose(self, plan_choose):
        self.plan_choose = plan_choose

    def get_features(self):
        return self.features

    def normal_distribution(self, x, mean, sigma):
        return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)

    def action_space(self):
        return self.action_dim

    def create_plan(self):
        # 0 Sin, 1 Gaussian, 2 Step
        self.one_hot = self.plan_choose

        if self.plan_choose == 0:
            x = np.arange(self.plan_width)
            y = np.round(10 * np.sin(2 * np.pi / self.plan_width * x) + self.plan_height)
        elif self.plan_choose == 1:
            mean1, sigma1 = 0, 3
            x = np.linspace(mean1 - 6 * sigma1, mean1 + 6 * sigma1, 30)
            y = np.round(self.normal_distribution(x, mean1, sigma1) * 100 + 17)
        elif self.plan_choose == 2:
            y = np.ones(30, ) * 15
            y[0:5] = 25
            y[10:15] = 25
            y[20:25] = 25
        else:
            raise ValueError('0: Sin, 1: Gaussian, 2: Step')

        area = sum(y)
        return y, area

    def clip_position(self, position):
        if position <= self.HALF_WINDOW_SIZE:
            position = self.HALF_WINDOW_SIZE
        elif position >= self.plan_width + self.HALF_WINDOW_SIZE - 1:
            position = self.plan_width + self.HALF_WINDOW_SIZE - 1
        return position

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(1, self.HALF_WINDOW_SIZE * 2 + 3))

    def reset(self):
        self._reset_buffer()
        self.one_hot = None
        self.plan, self.total_brick = self.create_plan()

        self.environment_memory = np.zeros((1, self.environment_width))
        self.environment_memory[:, :self.HALF_WINDOW_SIZE] = -1
        self.environment_memory[:, -self.HALF_WINDOW_SIZE:] = -1   # e.g.: [-1, -1, 0, 0, 0, 0, 0, 0, -1, -1]

        self.count_brick = 0
        self.count_step = 0

        self.brick_memory = []
        self.position_memory = []
        self.observation = None
        initial_position = self.HALF_WINDOW_SIZE

        self.position_memory.append(initial_position)
        self.brick_memory.append([-1, -1])

        return np.hstack((
            self.environment_memory[:, initial_position - self.HALF_WINDOW_SIZE : initial_position + self.HALF_WINDOW_SIZE + 1],
            np.array([[self.count_brick]]),
            np.array([[self.count_step]])))

    def step(self, action):
        if self.uniform_step:
            self.step_size = 1
        else:
            self.step_size = np.random.randint(1, 4)

        self.count_step += 1
        position = -1

        if action == 0:
            position = self.position_memory[-1] - self.step_size
            position = self.clip_position(position)
            self.position_memory.append(position)
            self.brick_memory.append([-1, -1])

        if action == 1:
            position = self.position_memory[-1] + self.step_size
            position = self.clip_position(position)
            self.position_memory.append(position)
            self.brick_memory.append([-1, -1])

        if action == 2:
            self.count_brick += 1
            position = self.position_memory[-1]
            self.position_memory.append(position)
            # if self.environment_memory[0, position] < self.environment_height:
            self.environment_memory[0, position] += 1
            self.brick_memory.append([position, self.environment_memory[0, position]])

            done = bool(self.count_brick > self.total_brick)

            if done:
                observation = np.hstack(
                    (self.environment_memory[:, position - self.HALF_WINDOW_SIZE: position + self.HALF_WINDOW_SIZE + 1],
                     np.array([[self.count_brick]]),
                     np.array([[self.count_step]])))
                # observation = torch.Tensor(observation)
                reward = 0.0

                # self.state_buffer.append(observation)
                return observation, reward, done
            else:
                done = bool(self.count_step >= self.total_step)
                if self.environment_memory[0, position] > self.plan[position - self.HALF_WINDOW_SIZE]:
                    reward = -1.0
                elif self.environment_memory[0, position] == self.plan[position - self.HALF_WINDOW_SIZE]:
                    reward = 10.0
                else:
                    reward = 1.0

                observation = np.hstack(
                    (self.environment_memory[:, position - self.HALF_WINDOW_SIZE:position + self.HALF_WINDOW_SIZE + 1],
                     np.array([[self.count_brick]]),
                     np.array([[self.count_step]])))

                # observation = torch.Tensor(observation)
                # self.state_buffer.append(observation)
                return observation, reward, done

        done = bool(self.count_step >= self.total_step)
        observation = np.hstack(
            (self.environment_memory[:, position - self.HALF_WINDOW_SIZE:position + self.HALF_WINDOW_SIZE + 1],
             np.array([[self.count_brick]]),
             np.array([[self.count_step]])))
        reward = 0

        # observation = torch.Tensor(observation)
        # self.state_buffer.append(observation)
        return observation, reward, done

    def render(self, ax, args, environment_memory, plan, highest_iou, count_step,
               count_brick, datetime, iou_min=None, iou_average=None, iter_times=10):

        x = np.arange(30)

        ax.set_xlim((-1, 30))
        ax.set_ylim((0, 50))
        ax.plot(x, plan)

        print("\tENV STATE: ", environment_memory)

        ax.title.set_text('step=%d,used_paint=%d,IOU=%.3f' % (count_step, count_brick, highest_iou))
        ax.plot(x, plan, color='b')
        ax.bar(x, environment_memory, color='r')

        ax.bar(x, environment_memory - 1, color='r')

        plt.text(12, 46, 'Iou_min_iter_%d = %.3f' % (iter_times, iou_min), color='red', fontsize=10)
        plt.text(12, 48, 'Iou_average_iter_%d = %.3f' % (iter_times, iou_average), color='blue', fontsize=10)

        save_plot(args, datetime)

        # plt.show()
        plt.close()

    def _iou(self):
        intersection = 0
        union = 0
        for i in range(self.plan_width):
          intersection += min(self.environment_memory[0, i + self.HALF_WINDOW_SIZE], self.plan[i])
          union += max(self.environment_memory[0, i + self.HALF_WINDOW_SIZE], self.plan[i])
        return intersection / union

class Env1DDynamic(gym.Env):
    def __init__(self, args):
        self.plan_width = 30
        self.plan_height = 20

        self.plan_choose = (args.plan_choose if args.plan_choose is not None else 0)
        self.environment_height = 100
        self.environment_memory = None
        self.count_brick = None
        self.brick_memory = None
        self.HALF_WINDOW_SIZE = args.half_window_size
        self.environment_width = self.plan_width + 2 * self.HALF_WINDOW_SIZE
        self.wall = np.ones((1, 2)) * (-1)
        self.position_memory = None
        self.observation = None
        self.count_step = 0
        self.total_step = 750
        self.plan, self.total_brick = self.create_plan()
        self.action_dim = 3
        self.state_dim = args.half_window_size * 2 + 3
        self.window = args.history_length
        self.uniform_step = args.uniform_step

        self.step_size = 1

        self.state_buffer = deque([], maxlen=self.window)
        self.features = 7 - 1

    def get_features(self):
        return self.features

    def normal_distribution(self, x, mean, sigma):
        return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)

    def action_space(self):
        return self.action_dim

    def create_plan(self):
        k_1 = np.random.uniform(3, 12)
        k_2 = np.random.randint(1, 4)
        phase = np.random.uniform(-1, 1) * np.pi
        one_hot = [k_1, k_2, phase]
        self.one_hot = one_hot

        x = np.arange(self.plan_width)
        y = np.round((k_1 * np.sin(2 * np.pi / self.plan_width * (k_2 * x + phase)) + self.plan_height))
        area = sum(y)

        return y, area

    def clip_position(self, position):
        if position <= self.HALF_WINDOW_SIZE:
            position = self.HALF_WINDOW_SIZE
        elif position >= self.plan_width + self.HALF_WINDOW_SIZE - 1:
            position = self.plan_width + self.HALF_WINDOW_SIZE - 1
        return position

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(1, self.HALF_WINDOW_SIZE * 2 + 3))

    def reset(self):
        self._reset_buffer()
        self.one_hot = None
        self.plan, self.total_brick = self.create_plan()

        self.environment_memory = np.zeros((1, self.environment_width))
        self.environment_memory[:, :self.HALF_WINDOW_SIZE] = -1
        self.environment_memory[:, -self.HALF_WINDOW_SIZE:] = -1   # e.g.: [-1, -1, 0, 0, 0, 0, 0, 0, -1, -1]

        self.count_brick = 0
        self.count_step = 0

        self.brick_memory = []
        self.position_memory = []
        self.observation = None
        initial_position = self.HALF_WINDOW_SIZE

        self.position_memory.append(initial_position)
        self.brick_memory.append([-1, -1])

        return np.hstack((self.environment_memory[:,
                          initial_position - self.HALF_WINDOW_SIZE:initial_position + self.HALF_WINDOW_SIZE + 1],
                          np.array([[self.count_brick]]), np.array([[self.count_step]])))

    def step(self, action):
        self.count_step += 1

        if self.uniform_step:
            self.step_size = 1
        else:
            self.step_size = np.random.randint(1, 4)

        if action == 0:
            position = self.position_memory[-1] - self.step_size
            position = self.clip_position(position)
            self.position_memory.append(position)
            self.brick_memory.append([-1, -1])

        if action == 1:
            position = self.position_memory[-1] + self.step_size
            position = self.clip_position(position)
            self.position_memory.append(position)
            self.brick_memory.append([-1, -1])

        if action == 2:
            self.count_brick += 1
            position = self.position_memory[-1]
            self.position_memory.append(position)
            self.environment_memory[0, position] += 1
            self.brick_memory.append([position, self.environment_memory[0, position]])

            done = bool(self.count_brick > self.total_brick)

            if done:
                observation = np.hstack(
                    (self.environment_memory[:, position - self.HALF_WINDOW_SIZE: position + self.HALF_WINDOW_SIZE + 1],
                     np.array([[self.count_brick]]),
                     np.array([[self.count_step]])))
                reward = 0.0
                return observation, reward, done
            else:
                done = bool(self.count_step >= self.total_step)
                if self.environment_memory[0, position] > self.plan[position - self.HALF_WINDOW_SIZE]:
                    reward = -1.0
                elif self.environment_memory[0, position] == self.plan[position - self.HALF_WINDOW_SIZE]:
                    reward = 10.0
                else:
                    reward = 1.0

                observation = np.hstack(
                    (self.environment_memory[:, position - self.HALF_WINDOW_SIZE:position + self.HALF_WINDOW_SIZE + 1],
                     np.array([[self.count_brick]]),
                     np.array([[self.count_step]])))

                return observation, reward, done

        done = bool(self.count_step >= self.total_step)
        observation = np.hstack(
            (self.environment_memory[:, position - self.HALF_WINDOW_SIZE:position + self.HALF_WINDOW_SIZE + 1],
             np.array([[self.count_brick]]),
             np.array([[self.count_step]])))
        reward = 0

        return observation, reward, done

    def render(self, ax, args, environment_memory, plan, highest_iou, count_step,
               count_brick, datetime, iou_min=None, iou_average=None, iter_times=10):

        x = np.arange(30)

        ax.set_xlim((-1, 30))
        ax.set_ylim((0, 50))
        ax.plot(x, plan)

        print("\tENV STATE: ", environment_memory)

        ax.title.set_text('step=%d,used_paint=%d,IOU=%.3f' % (count_step, count_brick, highest_iou))
        ax.plot(x, plan, color='b')
        ax.bar(x, environment_memory, color='r')

        ax.bar(x, environment_memory - 1, color='r')

        plt.text(12, 46, 'Iou_min_iter_%d = %.3f' % (iter_times, iou_min), color='red', fontsize=10)
        plt.text(12, 48, 'Iou_average_iter_%d = %.3f' % (iter_times, iou_average), color='blue', fontsize=10)

        save_plot(args, datetime)

        plt.close()

    def _iou(self):
        intersection = 0
        union = 0
        for i in range(self.plan_width):
          intersection += min(self.environment_memory[0, i + self.HALF_WINDOW_SIZE], self.plan[i])
          union += max(self.environment_memory[0, i + self.HALF_WINDOW_SIZE], self.plan[i])
        return intersection / union

class Env1DDynamic_Validation(gym.Env):
    def __init__(self, args):
        self.plan_width = 30
        self.plan_height = 20

        self.plan_choose = (args.plan_choose if args.plan_choose is not None else 0)
        self.environment_height = 100
        self.environment_memory = None
        self.count_brick = None
        self.brick_memory = None
        self.HALF_WINDOW_SIZE = 2
        self.environment_width = self.plan_width + 2 * self.HALF_WINDOW_SIZE
        self.wall = np.ones((1, 2)) * (-1)
        self.position_memory = None
        self.observation = None
        self.count_step = 0
        self.total_step = 750
        self.plan, self.total_brick = self.create_plan()
        self.action_dim = 3
        self.state_dim = args.half_window_size * 2 + 3
        self.window = args.history_length
        self.uniform_step = args.uniform_step

        self.step_size = 1

        self.state_buffer = deque([], maxlen=self.window)
        self.features = 7 - 1

    def set_plan_choose(self, plan_choose):
        self.plan_choose = plan_choose

    def get_features(self):
        return self.features

    def normal_distribution(self, x, mean, sigma):
        return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)

    def action_space(self):
        return self.action_dim

    def create_plan(self):
        test_set = [[3,1,-np.pi],[4,2,-0.5*np.pi],[5,3,-0.5*np.pi],
                    [6,1,np.pi],[7,2,-np.pi],[7,3,-0.5*np.pi],
                    [8,1,0.5*np.pi],[9,2,np.pi],[10,3,-np.pi],[11,1,-0.5*np.pi]]

        plan_width = self.plan_width

        one_hot = test_set[self.plan_choose]
        self.one_hot = one_hot

        x = np.arange(plan_width)
        y = np.round((one_hot[0] * np.sin(2 * np.pi / plan_width * (one_hot[1] * x + one_hot[2])) + self.plan_height))
        area = sum(y)

        return y, area

    def clip_position(self, position):
        if position <= self.HALF_WINDOW_SIZE:
            position = self.HALF_WINDOW_SIZE
        elif position >= self.plan_width + self.HALF_WINDOW_SIZE - 1:
            position = self.plan_width + self.HALF_WINDOW_SIZE - 1
        return position

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(1, self.HALF_WINDOW_SIZE * 2 + 3))

    def reset(self):
        self._reset_buffer()
        self.one_hot = None
        self.plan, self.total_brick = self.create_plan()

        self.environment_memory = np.zeros((1, self.environment_width))
        self.environment_memory[:, :self.HALF_WINDOW_SIZE] = -1
        self.environment_memory[:, -self.HALF_WINDOW_SIZE:] = -1   # e.g.: [-1, -1, 0, 0, 0, 0, 0, 0, -1, -1]

        self.count_brick = 0
        self.count_step = 0

        self.brick_memory = []
        self.position_memory = []
        self.observation = None
        initial_position = self.HALF_WINDOW_SIZE

        self.position_memory.append(initial_position)
        self.brick_memory.append([-1, -1])

        return np.hstack((self.environment_memory[:,
                          initial_position - self.HALF_WINDOW_SIZE:initial_position + self.HALF_WINDOW_SIZE + 1],
                          np.array([[self.count_brick]]), np.array([[self.count_step]])))

    def step(self, action):
        self.count_step += 1

        if self.uniform_step:
            self.step_size = 1
        else:
            self.step_size = np.random.randint(1, 4)

        if action == 0:
            position = self.position_memory[-1] - self.step_size
            position = self.clip_position(position)
            self.position_memory.append(position)
            self.brick_memory.append([-1, -1])

        if action == 1:
            position = self.position_memory[-1] + self.step_size
            position = self.clip_position(position)
            self.position_memory.append(position)
            self.brick_memory.append([-1, -1])

        if action == 2:
            self.count_brick += 1
            position = self.position_memory[-1]
            self.position_memory.append(position)
            self.environment_memory[0, position] += 1
            self.brick_memory.append([position, self.environment_memory[0, position]])

            done = bool(self.count_brick > self.total_brick)

            if done:
                observation = np.hstack(
                    (self.environment_memory[:, position - self.HALF_WINDOW_SIZE: position + self.HALF_WINDOW_SIZE + 1],
                     np.array([[self.count_brick]]),
                     np.array([[self.count_step]])))
                # observation = torch.Tensor(observation)
                reward = 0.0

                # self.state_buffer.append(observation)
                return observation, reward, done
            else:
                done = bool(self.count_step >= self.total_step)
                if self.environment_memory[0, position] > self.plan[position - self.HALF_WINDOW_SIZE]:
                    reward = -1.0
                elif self.environment_memory[0, position] == self.plan[position - self.HALF_WINDOW_SIZE]:
                    reward = 10.0
                else:
                    reward = 1.0

                observation = np.hstack(
                    (self.environment_memory[:, position - self.HALF_WINDOW_SIZE:position + self.HALF_WINDOW_SIZE + 1],
                     np.array([[self.count_brick]]),
                     np.array([[self.count_step]])))

                # observation = torch.Tensor(observation)
                # self.state_buffer.append(observation)
                return observation, reward, done

        done = bool(self.count_step >= self.total_step)
        observation = np.hstack(
            (self.environment_memory[:, position - self.HALF_WINDOW_SIZE:position + self.HALF_WINDOW_SIZE + 1],
             np.array([[self.count_brick]]),
             np.array([[self.count_step]])))
        reward = 0

        # observation = torch.Tensor(observation)
        # self.state_buffer.append(observation)
        return observation, reward, done

    def render(self, ax, args, environment_memory, plan, highest_iou, count_step,
               count_brick, datetime, iou_min=None, iou_average=None, iter_times=10):

        ax.clear()
        ax.set_xlabel('X-axis')
        ax.set_xlim(-1, 30)
        ax.set_ylabel('Y-axis')
        ax.set_ylim(0, 50)
        x = np.arange(self.plan_width)
        IOU = self._iou()
        plan = self.plan

        # print(environment_memory.shape)
        # env_memory = environment_memory[self.HALF_WINDOW_SIZE:self.plan_width + self.HALF_WINDOW_SIZE]
        IOU = highest_iou
        step = count_step
        brick = count_brick
        # else:
        #     env_memory = self.environment_memory[0][self.HALF_WINDOW_SIZE:self.plan_width + self.HALF_WINDOW_SIZE]
        #     step = self.count_step
        #     brick = self.count_brick

        ax.title.set_text('step=%d,used_paint=%d,IOU=%.3f' % (step, brick, IOU))
        ax.plot(x, plan, color='b')

        # print(x.shape, env_memory.shape)
        ax.bar(x, environment_memory, color='r')


        if iou_min == None:
            iou_min = IOU
        if iou_average == None:
            iou_average = IOU
        plt.text(0, 48, 'a=%d,b=%d,c=%.2f' % (self.one_hot[0], self.one_hot[1],self.one_hot[2]), color='g', fontsize=10)
        plt.text(12, 46, 'Iou_min_iter_%d = %.3f' % (iter_times, iou_min), color='red', fontsize=10)
        plt.text(12, 48, 'Iou_average_iter_%d = %.3f' % (iter_times, iou_average), color='blue', fontsize=10)

        save_plot(args, datetime)
        plt.draw()

    def _iou(self):
        intersection = 0
        union = 0
        for i in range(self.plan_width):
          intersection += min(self.environment_memory[0, i + self.HALF_WINDOW_SIZE], self.plan[i])
          union += max(self.environment_memory[0, i + self.HALF_WINDOW_SIZE], self.plan[i])
        return intersection / union