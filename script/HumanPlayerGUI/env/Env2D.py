from collections import deque

import cv2
import gym
import numpy as np
from matplotlib import patches


class Env2DStatic(gym.Env):
    def __init__(self, args):
        self.plan_width = 20
        self.plan_height = 20

        self.plan_choose = (args.plan_choose if args.plan_choose is not None else 0)

        self.environment_memory = None
        self.count_brick = None
        self.brick_memory = None
        self.HALF_WINDOW_SIZE = 3
        self.environment_width = self.plan_width + 2 * self.HALF_WINDOW_SIZE
        self.environment_height = self.plan_height + 2 * self.HALF_WINDOW_SIZE

        self.position_memory = None
        self.observation = None
        self.count_step = 0
        self.total_step = 600
        self.plan, self.total_brick = self.create_plan()
        self.input_plan = None

        self.action_dim = 5
        self.state_dim = (args.half_window_size * 2 + 1) ** 2 + 2
        self.window = args.history_length
        self.uniform_step = args.uniform_step
        self.step_size = 1

        self.features = 51 - 1

    def set_plan_choose(self, plan_choose):
        self.plan_choose = plan_choose

    def get_features(self):
        return self.features

    def action_space(self):
        return self.action_dim

    def create_plan(self):
        out_radius = 7
        if self.plan_choose == 0:
            in_radius = 0
        elif self.plan_choose == 1:
            out_radius = 8
            in_radius = 7
        else:
            raise ValueError('0: Dense circle, 1: Sparse circle')

        plan = np.zeros((self.environment_height, self.environment_width))
        a = np.array([12.5, 12.5])
        circle = patches.CirclePolygon(a, out_radius)
        circle2 = patches.CirclePolygon(a, in_radius)
        for i in range(len(plan)):
            for j in range(len(plan[0])):
                a = np.array([i, j])
                if circle.contains_point(a) and not circle2.contains_point(a):
                    plan[i][j] = 1

            total_area = sum(sum(plan))

        area=sum(sum(plan))

        return plan, area

    def observation_(self, position):
        observation = self.environment_memory[
                      position[0] - self.HALF_WINDOW_SIZE : position[0] + self.HALF_WINDOW_SIZE + 1,
                      position[1] - self.HALF_WINDOW_SIZE : position[1] + self.HALF_WINDOW_SIZE + 1]

        return observation.flatten().reshape(1, -1)

    def clip_position(self, position):
        if position[0] <= self.HALF_WINDOW_SIZE:
            position[0] = self.HALF_WINDOW_SIZE
        if position[1] <= self.HALF_WINDOW_SIZE:
            position[1] = self.HALF_WINDOW_SIZE
        if position[0] >= self.plan_width+self.HALF_WINDOW_SIZE - 1:
            position[0] = self.plan_width+self.HALF_WINDOW_SIZE - 1
        if position[1] >= self.plan_width+self.HALF_WINDOW_SIZE - 1:
            position[1] = self.plan_width+self.HALF_WINDOW_SIZE -1
        return position

    def reset(self):
        self.plan, self.total_brick = self.create_plan()

        if self.total_brick<30:
            self.total_brick=30

        self.input_plan = self.plan[
                          self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_height,
                          self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_width]
        self.environment_memory = np.zeros((self.environment_height, self.environment_width))
        self.environment_memory[:, :self.HALF_WINDOW_SIZE] = -1
        self.environment_memory[:, -self.HALF_WINDOW_SIZE:] = -1
        self.environment_memory[:self.HALF_WINDOW_SIZE, :] = -1
        self.environment_memory[-self.HALF_WINDOW_SIZE:, :] = -1

        self.count_brick = 0

        self.position_memory = []
        self.observation = None
        self.count_step = 0
        initial_position = [self.HALF_WINDOW_SIZE, self.HALF_WINDOW_SIZE]

        self.position_memory.append(initial_position)
        #
        return np.hstack(
            (self.observation_(initial_position),
             np.array([[self.count_brick]]),
             np.array([[self.count_step]])))

    def step(self, action):
        if self.uniform_step:
            self.step_size = 1
        else:
            self.step_size = np.random.randint(1, 4)

        self.count_step += 1

        ### 0 left, 1 right, 2 up, 3 down
        if action == 0:
            position = [self.position_memory[-1][0], self.position_memory[-1][1] - self.step_size]
            position = self.clip_position(position)
            self.position_memory.append(position)
        if action == 1:
            position = [self.position_memory[-1][0], self.position_memory[-1][1] + self.step_size]
            position = self.clip_position(position)
            self.position_memory.append(position)
        if action == 2:
            position = [self.position_memory[-1][0] + self.step_size, self.position_memory[-1][1]]
            position = self.clip_position(position)
            self.position_memory.append(position)
        if action == 3:
            position = [self.position_memory[-1][0] - self.step_size, self.position_memory[-1][1]]
            position = self.clip_position(position)
            self.position_memory.append(position)

        #######    4   drop
        if action == 4:
            self.count_brick += 1
            position = self.position_memory[-1]
            self.position_memory.append(position)

            self.environment_memory[position[0], position[1]] += 1.0

            done = bool(self.count_brick > self.total_brick)

            if done:
                if self.environment_memory[position[0],position[1]]>1:
                    self.environment_memory[position[0],position[1]]=1.0

                observation = np.hstack(
                    (self.observation_(position),
                     np.array([[self.count_brick]]),
                     np.array([[self.count_step]])))

                reward = 0.0

                return observation, reward, done
            else:
                if self.environment_memory[position[0], position[1]] > self.plan[position[0], position[1]]:
                    reward = 0
                elif self.environment_memory[position[0], position[1]] == self.plan[position[0], position[1]]:
                    reward = 5.0

                if self.environment_memory[position[0], position[1]] > 1.0:
                    self.environment_memory[position[0], position[1]] = 1.0

                observation = np.hstack(
                    (self.observation_(position),
                     np.array([[self.count_brick]]),
                     np.array([[self.count_step]])))

                return observation, reward, done

        done = bool(self.count_step >= self.total_step)
        observation = np.hstack(
            (self.observation_(position),
             np.array([[self.count_brick]]),
             np.array([[self.count_step]])))
        reward = 0

        return observation, reward, done

    def _iou(self):
        component1 = self.plan[
                     self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_height,
                     self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_width]\
            .astype(bool)

        component2 = self.environment_memory[
                     self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_height,
                     self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_width]\
            .astype(bool)

        overlap = component1 * component2 # Logical AND
        union = component1 + component2 # Logical OR

        return overlap.sum()/float(union.sum())

class Env2DDynamic(gym.Env):
    def __init__(self, args):
        self.plan_width = 20
        self.plan_height = 20

        # plan_choose: 0 Dense triangle, 1 Sparse triangle
        self.plan_choose = (args.plan_choose if args.plan_choose is not None else 0)

        self.environment_memory = None
        self.count_brick = None
        self.brick_memory = None
        self.HALF_WINDOW_SIZE = 3
        self.environment_width = self.plan_width + 2 * self.HALF_WINDOW_SIZE
        self.environment_height = self.plan_height + 2 * self.HALF_WINDOW_SIZE

        self.position_memory = None
        self.observation = None
        self.count_step = 0
        self.total_step = 600
        self.plan, self.total_brick = self.create_plan()
        self.input_plan = None

        self.action_dim = 5
        self.state_dim = (args.half_window_size * 2 + 1) ** 2 + 2
        self.window = args.history_length
        self.uniform_step = args.uniform_step
        self.step_size = 1

        self.features = 51 - 1

    def set_plan_choose(self, plan_choose):
        self.plan_choose = plan_choose

    def get_features(self):
        return self.features

    def action_space(self):
        return self.action_dim

    def create_plan(self):
        if not(self.plan_choose == 1 or self.plan_choose == 0 ):
            raise ValueError(' 0: Dense triangle, 1: Sparse triangle')

        plan = np.zeros((self.environment_height, self.environment_width))
        area = [50, 20]
        total_area = 0
        while total_area <= area[self.plan_choose]:
            x = np.random.randint(0, self.plan_width, size=3)
            y = np.random.randint(0, self.plan_height, size=3)

            img_rgb = np.ones((self.plan_height, self.plan_width, 3), np.uint8) * 255
            vertices = np.array([[x[0], y[0]], [x[1], y[1]], [x[2], y[2]]], np.int32)
            pts = vertices.reshape((-1, 1, 2))
            cv2.polylines(img_rgb, [pts], isClosed=True, color=(0, 0, 0))
            if self.plan_choose == 1:
                cv2.fillPoly(img_rgb, [pts], color=(0, 0, 0))

            plan[
            self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_height,
            self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_width] = 1 - img_rgb[:, :, 0] / 255

            total_area = sum(sum(plan))

        return plan, total_area

    def observation_(self, position):
        observation = self.environment_memory[
                      position[0] - self.HALF_WINDOW_SIZE : position[0] + self.HALF_WINDOW_SIZE + 1,
                      position[1] - self.HALF_WINDOW_SIZE : position[1] + self.HALF_WINDOW_SIZE + 1]

        return observation.flatten().reshape(1, -1)

    def clip_position(self, position):
        if position[0] <= self.HALF_WINDOW_SIZE:
            position[0] = self.HALF_WINDOW_SIZE
        if position[1] <= self.HALF_WINDOW_SIZE:
            position[1] = self.HALF_WINDOW_SIZE
        if position[0] >= self.plan_width+self.HALF_WINDOW_SIZE - 1:
            position[0] = self.plan_width+self.HALF_WINDOW_SIZE - 1
        if position[1] >= self.plan_width+self.HALF_WINDOW_SIZE - 1:
            position[1] = self.plan_width+self.HALF_WINDOW_SIZE -1
        return position

    def reset(self):
        self.plan, self.total_brick = self.create_plan()

        if self.total_brick<30:
            self.total_brick=30

        self.input_plan = self.plan[
                          self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_height,
                          self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_width]
        self.environment_memory = np.zeros((self.environment_height, self.environment_width))
        self.environment_memory[:, :self.HALF_WINDOW_SIZE] = -1
        self.environment_memory[:, -self.HALF_WINDOW_SIZE:] = -1
        self.environment_memory[:self.HALF_WINDOW_SIZE, :] = -1
        self.environment_memory[-self.HALF_WINDOW_SIZE:, :] = -1

        self.count_brick = 0

        self.position_memory = []
        self.observation = None
        self.count_step = 0
        initial_position = [self.HALF_WINDOW_SIZE, self.HALF_WINDOW_SIZE]

        self.position_memory.append(initial_position)
        #
        return np.hstack(
            (self.observation_(initial_position),
             np.array([[self.count_brick]]),
             np.array([[self.count_step]])))

    def step(self, action):
        if self.uniform_step:
            self.step_size = 1
        else:
            self.step_size = np.random.randint(1, 4)

        self.count_step += 1

        ### 0 left, 1 right, 2 up, 3 down
        if action == 0:
            position = [self.position_memory[-1][0], self.position_memory[-1][1] - self.step_size]
            position = self.clip_position(position)
            self.position_memory.append(position)
        if action == 1:
            position = [self.position_memory[-1][0], self.position_memory[-1][1] + self.step_size]
            position = self.clip_position(position)
            self.position_memory.append(position)
        if action == 2:
            position = [self.position_memory[-1][0] + self.step_size, self.position_memory[-1][1]]
            position = self.clip_position(position)
            self.position_memory.append(position)
        if action == 3:
            position = [self.position_memory[-1][0] - self.step_size, self.position_memory[-1][1]]
            position = self.clip_position(position)
            self.position_memory.append(position)

        #######    4   drop
        if action == 4:
            self.count_brick += 1
            position = self.position_memory[-1]
            self.position_memory.append(position)

            self.environment_memory[position[0], position[1]] += 1.0

            done = bool(self.count_brick > self.total_brick)

            if done:
                if self.environment_memory[position[0],position[1]]>1:
                    self.environment_memory[position[0],position[1]]=1.0

                observation = np.hstack(
                    (self.observation_(position),
                     np.array([[self.count_brick]]),
                     np.array([[self.count_step]])))

                reward = 0.0

                return observation, reward, done
            else:
                if self.environment_memory[position[0], position[1]] > self.plan[position[0], position[1]]:
                    reward = 0
                elif self.environment_memory[position[0], position[1]] == self.plan[position[0], position[1]]:
                    reward = 5.0

                if self.environment_memory[position[0], position[1]] > 1.0:
                    self.environment_memory[position[0], position[1]] = 1.0

                observation = np.hstack(
                    (self.observation_(position),
                     np.array([[self.count_brick]]),
                     np.array([[self.count_step]])))

                return observation, reward, done

        done = bool(self.count_step >= self.total_step)
        observation = np.hstack(
            (self.observation_(position),
             np.array([[self.count_brick]]),
             np.array([[self.count_step]])))
        reward = 0

        return observation, reward, done

    def _iou(self):
        component1 = self.plan[
                     self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_height,
                     self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_width]\
            .astype(bool)

        component2 = self.environment_memory[
                     self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_height,
                     self.HALF_WINDOW_SIZE : self.HALF_WINDOW_SIZE + self.plan_width]\
            .astype(bool)

        overlap = component1 * component2 # Logical AND
        union = component1 + component2 # Logical OR

        return overlap.sum()/float(union.sum())