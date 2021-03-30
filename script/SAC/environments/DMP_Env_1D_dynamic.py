import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
import torch
from gym import spaces


class deep_mobile_printing_1d1r(gym.Env):
    def __init__(self):

        self.step_size = 1
        self.plan_width = 30  # plan parameters: width, height, base_line
        self.plan_height = 20
        self.environment_height = 100
        self.environment_memory = None
        self.conut_brick = None
        self.brick_memory = None
        self.HALF_WINDOW_SIZE = 2
        self.environment_width = self.plan_width + 2 * self.HALF_WINDOW_SIZE
        self.wall = np.ones((1, 2)) * (-1)
        self.position_memory = None
        self.observation = None
        self.count_step = 0
        self.total_step = 750
        self.plan = None
        self.total_brick = 0
        self.one_hot = None

        self.action_dim = 3
        self.state_dim = self.HALF_WINDOW_SIZE * 2 + 1 + 2 + self.plan_width

    def create_plan(self):
        plan_width = self.plan_width

        k_1 = np.random.uniform(3, 12)
        k_2 = np.random.randint(1, 4)
        phase = np.random.uniform(-1, 1) * np.pi
        one_hot = [k_1, k_2, phase]
        self.one_hot = one_hot

        x = np.arange(plan_width)
        y = np.round((k_1 * np.sin(2 * np.pi / plan_width * (k_2 * x + phase)) + self.plan_height))
        area = sum(y)

        return y, area

    def clip_position(self, position):
        if position <= self.HALF_WINDOW_SIZE:
            position = self.HALF_WINDOW_SIZE
        elif position >= self.plan_width + self.HALF_WINDOW_SIZE - 1:
            position = self.plan_width + self.HALF_WINDOW_SIZE - 1
        else:
            position = position
        return position

    def reset(self):
        self.one_hot = None
        self.plan, self.total_brick = self.create_plan()
        self.environment_memory = np.zeros((1, self.environment_width))
        self.environment_memory[:, :self.HALF_WINDOW_SIZE] = -1
        self.environment_memory[:, -self.HALF_WINDOW_SIZE:] = -1
        self.conut_brick = 0
        self.brick_memory = []
        self.position_memory = []
        self.observation = None
        self.count_step = 0
        # initial_position = np.random.randint(self.HALF_WINDOW_SIZE,self.plan_width+self.HALF_WINDOW_SIZE)
        initial_position = self.HALF_WINDOW_SIZE

        self.position_memory.append(initial_position)
        self.brick_memory.append([-1, -1])
        return np.hstack((self.environment_memory[:,
                          initial_position - self.HALF_WINDOW_SIZE:initial_position + self.HALF_WINDOW_SIZE + 1],
                          np.array([[self.conut_brick]]), np.array([[self.count_step]]), np.array([self.plan])))[0]

    def step(self, action):
        self.count_step += 1
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
            self.conut_brick += 1
            position = self.position_memory[-1]
            self.position_memory.append(position)
            self.environment_memory[0, position] += 1
            self.brick_memory.append([position, self.environment_memory[0, position]])
            done = bool(self.conut_brick > self.total_brick)
            if done:

                observation = np.hstack((self.environment_memory[:,
                                         position - self.HALF_WINDOW_SIZE:position + self.HALF_WINDOW_SIZE + 1],
                                         np.array([[self.conut_brick]]), np.array([[self.count_step]]),np.array([self.plan])))[0]
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

                observation = np.hstack((self.environment_memory[:,
                                         position - self.HALF_WINDOW_SIZE:position + self.HALF_WINDOW_SIZE + 1],
                                         np.array([[self.conut_brick]]), np.array([[self.count_step]]), np.array([self.plan])))[0]
                return observation, reward, done
        done = bool(self.count_step >= self.total_step)
        observation = np.hstack((self.environment_memory[:,
                                 position - self.HALF_WINDOW_SIZE:position + self.HALF_WINDOW_SIZE + 1],
                                 np.array([[self.conut_brick]]), np.array([[self.count_step]]), np.array([self.plan])))[0]
        reward = 0

        return observation, reward, done

    def iou(self):
        component1 = self.plan
        component2 = self.environment_memory[0][self.HALF_WINDOW_SIZE:self.plan_width + self.HALF_WINDOW_SIZE]

        area1 = sum(component1)
        area2 = sum(component2)
        k = 0
        for i in range(len(self.plan)):
            if component2[i] > component1[i]:
                j = component2[i] - component1[i]
                k += j
        area_cross = area2 - k
        iou = area_cross / (area1 + area2 - area_cross)
        return iou

    def render(self, axe, iou_min=None, iou_average=None, iter_times=100):
        axe.clear()
        axe.set_xlabel('X-axis')
        axe.set_xlim(-1, 30)
        axe.set_ylabel('Y-axis')
        axe.set_ylim(0, 50)
        x = np.arange(self.plan_width)
        IOU = self.iou()
        plan = self.plan
        env_memory = self.environment_memory[0][self.HALF_WINDOW_SIZE:self.plan_width + self.HALF_WINDOW_SIZE]
        axe.title.set_text('step=%d,used_paint=%d,IOU=%.3f' % (self.count_step, self.conut_brick, IOU))
        axe.plot(x, plan, color='b')

        axe.bar(x, env_memory - 1, color='r')

        if iou_min == None:
            iou_min = IOU
        if iou_average == None:
            iou_average = IOU
        plt.text(0, 48, 'a=%d,b=%d,c=%.2f' % (self.one_hot[0], self.one_hot[1], self.one_hot[2]), color='g',
                 fontsize=10)
        plt.text(12, 46, 'Iou_min_iter_%d = %.3f' % (iter_times, iou_min), color='red', fontsize=10)
        plt.text(12, 48, 'Iou_average_iter_%d = %.3f' % (iter_times, iou_average), color='blue', fontsize=10)

        plt.draw()


if __name__ == "__main__":

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    env = deep_mobile_printing_1d1r()
    observation = env.reset()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    print(env.total_brick)
    print(env.one_hot)
    step = env.total_step
    ax.clear()
    for i in range(step):
        action = np.random.randint(0, 3, (1,))
        observation, reward, done = env.step(action)
        env.render(ax)
        env.iou()
        plt.pause(0.01)
    plt.show()
