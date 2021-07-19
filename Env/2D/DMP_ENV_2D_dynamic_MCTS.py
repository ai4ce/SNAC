import numpy as np
import matplotlib.pyplot as plt
import gym
import cv2
from gym import spaces


class deep_mobile_printing_2d1r(gym.Env):
    def __init__(self, plan_choose=0):
        # plan_choose: 0 Dense triangle, 1 Sparse triangle
        self.step_size = 1
        self.plan_width = 20
        self.plan_height = 20
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
        self.plan = None
        self.input_plan = None
        self.total_brick = 0
        self.one_hot = None
        self.action_dim = 5
        self.action_space = spaces.Discrete(self.action_dim)
        self.state_dim = (2 * self.HALF_WINDOW_SIZE + 1) ** 2 + 2
        self.plan_choose = plan_choose

    def create_plan(self):
        if not (self.plan_choose == 1 or self.plan_choose == 0):
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

            plan[self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_height,
            self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_width] = 1 - img_rgb[:, :, 0] / 255
            total_area = sum(sum(plan))

        return plan, total_area

    def reset(self):
        self.plan, self.total_brick = self.create_plan()
        if self.total_brick < 30:
            self.total_brick = 30
        self.input_plan = self.plan[self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_height, \
                          self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_width]
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
        environment_memory=self.environment_memory.copy()
        count_brick=self.count_brick
        count_step=self.count_step
        self.state = (initial_position,environment_memory,count_brick,count_step)
        observation = np.hstack(
            (self.observation_(initial_position), np.array([[self.count_brick]]), np.array([[self.count_step]])))
        return self.state, observation

    def observation_(self, position):
        observation = self.environment_memory[
                      position[0] - self.HALF_WINDOW_SIZE:position[0] + self.HALF_WINDOW_SIZE + 1, \
                      position[1] - self.HALF_WINDOW_SIZE:position[1] + self.HALF_WINDOW_SIZE + 1]
        return observation.flatten().reshape(1, -1)
    
    def observation_transition(self, environment_memory, position):
        observation = environment_memory[
                      position[0] - self.HALF_WINDOW_SIZE:position[0] + self.HALF_WINDOW_SIZE + 1, \
                      position[1] - self.HALF_WINDOW_SIZE:position[1] + self.HALF_WINDOW_SIZE + 1]
        return observation.flatten().reshape(1, -1)

    def clip_position(self, position):
        if position[0] <= self.HALF_WINDOW_SIZE:
            position[0] = self.HALF_WINDOW_SIZE
        if position[1] <= self.HALF_WINDOW_SIZE:
            position[1] = self.HALF_WINDOW_SIZE
        if position[0] >= self.plan_width + self.HALF_WINDOW_SIZE - 1:
            position[0] = self.plan_width + self.HALF_WINDOW_SIZE - 1
        if position[1] >= self.plan_width + self.HALF_WINDOW_SIZE - 1:
            position[1] = self.plan_width + self.HALF_WINDOW_SIZE - 1
        return position

    def transition(self, state, action, is_model_dynamic=True):
        step_size = np.random.randint(1, 4)
        position, environment_memory, count_brick, count_step = state 
        count_step += 1
        
        # 0 left, 1 right, 2 up, 3 down
        if action == 0:
            position = [position[0], position[1] - step_size]
            position = self.clip_position(position)
        #
        if action == 1:
            position = [position[0], position[1] + step_size]
            position = self.clip_position(position)
        #
        if action == 2:
            position = [position[0] + step_size, position[1]]
            position = self.clip_position(position)

        if action == 3:
            position = [position[0] - step_size, position[1]]
            position = self.clip_position(position)

        # 4 drop
        if action == 4:
            count_brick += 1

            environment_memory[position[0], position[1]] += 1.0

            done = bool(count_brick >= self.total_brick)
            if done:
                if environment_memory[position[0], position[1]] > 1:
                    environment_memory[position[0], position[1]] = 1.0

                observation = np.hstack(
                    (self.observation_transition(environment_memory,position), np.array([[count_brick]]), np.array([[count_step]])))
                reward = 0.0
                state  = position, environment_memory, count_brick, count_step
                return state, observation, reward, done
            else:
                done = bool(count_step >= self.total_step)
                if environment_memory[position[0], position[1]] > self.plan[position[0], position[1]]:

                    reward = 0
                elif environment_memory[position[0], position[1]] == self.plan[position[0], position[1]]:
                    reward = 5.0
                if environment_memory[position[0], position[1]] > 1.0:
                    environment_memory[position[0], position[1]] = 1.0
                observation = np.hstack(
                    (self.observation_transition(environment_memory,position), np.array([[count_brick]]), np.array([[count_step]])))
                state  = position, environment_memory, count_brick, count_step
                return state, observation, reward, done
                

        done = bool(count_step >= self.total_step)
        observation = np.hstack(
            (self.observation_transition(environment_memory,position), np.array([[count_brick]]), np.array([[count_step]])))
        reward = 0
        state  = position, environment_memory, count_brick, count_step
        return state, observation, reward, done
            

    def step(self, action):
        self.count_step += 1
        self.step_size = np.random.randint(1, 4)

        # 0 left, 1 right, 2 up, 3 down
        if action == 0:
            position = [self.position_memory[-1][0], self.position_memory[-1][1] - self.step_size]
            position = self.clip_position(position)
            self.position_memory.append(position)
        #
        if action == 1:
            position = [self.position_memory[-1][0], self.position_memory[-1][1] + self.step_size]
            position = self.clip_position(position)
            self.position_memory.append(position)
        #
        if action == 2:
            position = [self.position_memory[-1][0] + self.step_size, self.position_memory[-1][1]]
            position = self.clip_position(position)
            self.position_memory.append(position)
        if action == 3:
            position = [self.position_memory[-1][0] - self.step_size, self.position_memory[-1][1]]
            position = self.clip_position(position)
            self.position_memory.append(position)

        # 4 drop
        if action == 4:
            self.count_brick += 1
            position = self.position_memory[-1]
            self.position_memory.append(position)

            self.environment_memory[position[0], position[1]] += 1.0

            done = bool(self.count_brick >= self.total_brick)
            if done:
                if self.environment_memory[position[0], position[1]] > 1:
                    self.environment_memory[position[0], position[1]] = 1.0

                observation = np.hstack(
                    (self.observation_(position), np.array([[self.count_brick]]), np.array([[self.count_step]])))

                environment_memory=self.environment_memory.copy()
                count_brick=self.count_brick
                count_step=self.count_step
                    
                self.state = (position,environment_memory,count_brick,count_step)
                reward = 0.0
                return self.state ,observation, reward, done
            else:
                done = bool(self.count_step >= self.total_step)
                if self.environment_memory[position[0], position[1]] > self.plan[position[0], position[1]]:

                    reward = 0
                elif self.environment_memory[position[0], position[1]] == self.plan[position[0], position[1]]:
                    reward = 5.0
                if self.environment_memory[position[0], position[1]] > 1.0:
                    self.environment_memory[position[0], position[1]] = 1.0
                observation = np.hstack(
                    (self.observation_(position), np.array([[self.count_brick]]), np.array([[self.count_step]])))

                environment_memory=self.environment_memory.copy()
                count_brick=self.count_brick
                count_step=self.count_step
                    
                self.state = (position,environment_memory,count_brick,count_step)
                return self.state, observation, reward, done

        done = bool(self.count_step >= self.total_step)
        observation = np.hstack(
            (self.observation_(position), np.array([[self.count_brick]]), np.array([[self.count_step]])))
        reward = 0
        environment_memory=self.environment_memory.copy()
        count_brick=self.count_brick
        count_step=self.count_step
                    
        self.state = (position,environment_memory,count_brick,count_step)
        return self.state, observation, reward, done
    def equality_operator(self, o1, o2):
        Equal=True 
        if not np.array_equal(o1,o2):
            return False
        return Equal

    def render(self, axe, iou_min=None, iou_average=None, iter_times=1):

        axe.clear()
        plan = self.plan[self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_height, \
               self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_width]
        env_memory = self.environment_memory[self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_height, \
                     self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_width]
        background = np.zeros((self.plan_height, self.plan_width))
        img = np.stack((env_memory, plan, background), axis=2)
        plt.imshow(img)
        plt.plot(self.position_memory[-1][1] - self.HALF_WINDOW_SIZE,
                 self.position_memory[-1][0] - self.HALF_WINDOW_SIZE, "*")
        ###  compute IOU
        component1 = self.plan[self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_height, \
                     self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_width].astype(bool)
        component2 = self.environment_memory[self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_height, \
                     self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_width].astype(bool)
        overlap = component1 * component2  # Logical AND
        union = component1 + component2  # Logical OR
        IOU = overlap.sum() / float(union.sum())

        # Add the patch to the Axes
        if iou_min == None:
            iou_min = IOU
        if iou_average == None:
            iou_average = IOU
        # axe.add_patch(rect)
        axe.title.set_text('step=%d,used_paint=%d,IOU=%.3f' % (self.count_step, self.count_brick, IOU))
        plt.text(0, 21.5, 'Iou_min_iter_%d = %.3f' % (iter_times, iou_min), color='red', fontsize=12)
        plt.text(0, 20.5, 'Iou_average_iter_%d = %.3f' % (iter_times, iou_average), color='blue', fontsize=12)
        axe.axis('off')

        plt.draw()


if __name__ == "__main__":
    env = deep_mobile_printing_2d1r(plan_choose=1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    total_reward = 0
    env.reset()
    env.render(ax)
    print("total_brick:", env.total_brick)
    while True:
        state, obs, reward, done = env.step(np.random.randint(5))
        env.render(ax)
        plt.pause(0.1)
        total_reward += reward
        if done:
            break

    print("reward:", total_reward)
    plt.show()
