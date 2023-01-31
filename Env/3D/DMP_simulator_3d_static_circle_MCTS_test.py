import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import matplotlib.patches as patches
import time

class deep_mobile_printing_3d1r(gym.Env):
    def __init__(self, plan_choose=1):

        self.step_size = 1

        self.plan_width = 20  # X Axis
        self.plan_height = 20  # Y Axis
        self.plan_length = 10  # Z Axis
        self.z = 6

        self.environment_memory = None
        self.count_brick = None
        self.brick_memory = None
        self.HALF_WINDOW_SIZE = 3
        self.environment_width = self.plan_width + 2 * self.HALF_WINDOW_SIZE
        self.environment_height = self.plan_height + 2 * self.HALF_WINDOW_SIZE
        self.environment_length = self.plan_length + 2 * self.HALF_WINDOW_SIZE
        self.start = None
        self.position_memory = None
        self.observation = None
        self.count_step = 0
        self.total_step = 1300
        self.plan = None
        self.input_plan = None
        self.total_brick = 0
        self.one_hot = None
        self.check = []
        self.action_dim = 8
        self.action_space = spaces.Discrete(self.action_dim)
        # up, down, left, right,
        # up_brick, down_brick, left_brick, right_brick
        self.state_dim = (2 * self.HALF_WINDOW_SIZE + 1) ** 2 + 2
        self.blank_size = 2  # use 0, 2, 4
        self.start = None
        self.plan_choose = plan_choose

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
        plan = plan * self.z
        total_area = total_area * self.z
        return plan, total_area

    def reset(self):
        self.plan, self.total_brick = self.create_plan()
        self.input_plan = self.plan[self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_height,
                          self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_width]
        self.environment_memory = np.zeros((self.environment_height, self.environment_width))
        self.environment_memory[:, :self.HALF_WINDOW_SIZE] = -1
        self.environment_memory[:, -self.HALF_WINDOW_SIZE:] = -1
        self.environment_memory[:self.HALF_WINDOW_SIZE, :] = -1
        self.environment_memory[-self.HALF_WINDOW_SIZE:, :] = -1
        self.check = []
        self.count_brick = 0
        self.step_size = 1
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

    def check_sur(self, position):
        # 0 left, 1 right, 2 up, 3 down
        check = [0, 0, 0, 0, 0, 0, 0, 0]

        observation = [self.environment_memory[position[0], position[1] - 1],
                       self.environment_memory[position[0], position[1] + 1],
                       self.environment_memory[position[0] + 1, position[1]],
                       self.environment_memory[position[0] - 1, position[1]]]
        for i in range(len(observation)):
            if observation[i] == -1:
                check[i] = 1
                check[i + 4] = 1
            elif observation[i] > 0:
                check[i] = 1
        return check
    def check_sur_trainsition(self, environment_memory,position):
        # 0 left, 1 right, 2 up, 3 down
        check = [0, 0, 0, 0, 0, 0, 0, 0]

        observation = [environment_memory[position[0], position[1] - 1],
                       environment_memory[position[0], position[1] + 1],
                       environment_memory[position[0] + 1, position[1]],
                       environment_memory[position[0] - 1, position[1]]]
        for i in range(len(observation)):
            if observation[i] == -1:
                check[i] = 1
                check[i + 4] = 1
            elif observation[i] > 0:
                check[i] = 1
        return check

    def move_step(self, position, action, step_size):
        move = 0
        if action == 0:
            for i in range(step_size):
                observation = self.environment_memory[position[0], position[1] - i - 1]
                if observation == 0:
                    move += 1
                else:
                    break
        elif action == 1:
            for i in range(step_size):
                observation = self.environment_memory[position[0], position[1] + i + 1]
                if observation == 0:
                    move += 1
                else:
                    break
        elif action == 2:
            for i in range(step_size):
                observation = self.environment_memory[position[0] + i + 1, position[1]]
                if observation == 0:
                    move += 1
                else:
                    break
        elif action == 3:
            for i in range(step_size):
                observation = self.environment_memory[position[0] - i - 1, position[1]]
                if observation == 0:
                    move += 1
                else:
                    break
        return move
    def move_step_transition(self, environment_memory, position, action, step_size):
        move = 0
        if action == 0:
            for i in range(step_size):
                observation = environment_memory[position[0], position[1] - i - 1]
                if observation == 0:
                    move += 1
                else:
                    break
        elif action == 1:
            for i in range(step_size):
                observation = environment_memory[position[0], position[1] + i + 1]
                if observation == 0:
                    move += 1
                else:
                    break
        elif action == 2:
            for i in range(step_size):
                observation = environment_memory[position[0] + i + 1, position[1]]
                if observation == 0:
                    move += 1
                else:
                    break
        elif action == 3:
            for i in range(step_size):
                observation = environment_memory[position[0] - i - 1, position[1]]
                if observation == 0:
                    move += 1
                else:
                    break
        return move

    def observation_(self, position):
        observation = self.environment_memory[
                      position[0] - self.HALF_WINDOW_SIZE:position[0] + self.HALF_WINDOW_SIZE + 1,
                      position[1] - self.HALF_WINDOW_SIZE:position[1] + self.HALF_WINDOW_SIZE + 1]
        return observation.flatten().reshape(1, -1)
    def observation_transition(self, environment_memory, position):
        observation = environment_memory[
                      position[0] - self.HALF_WINDOW_SIZE:position[0] + self.HALF_WINDOW_SIZE + 1,
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
        check = self.check_sur_trainsition(environment_memory,position)

        # moving: 0 left, 1 right, 2 up, 3 down
        if action == 0 and check[0] == 0:
            step_size = self.move_step_transition(environment_memory,[position[0], position[1]], action,
                                       step_size)
            position = [position[0], position[1] - step_size]
            position = self.clip_position(position)
        elif action == 1 and check[1] == 0:
            step_size = self.move_step_transition(environment_memory,[position[0], position[1]], action,
                                       step_size)
            position = [position[0], position[1] + step_size]
            position = self.clip_position(position)
        elif action == 2 and check[2] == 0:
            step_size = self.move_step_transition(environment_memory,[position[0], position[1]], action,
                                       step_size)
            position = [position[0] + step_size, position[1]]
            position = self.clip_position(position)
        elif action == 3 and check[3] == 0:
            step_size = self.move_step_transition(environment_memory,[position[0], position[1]], action,
                                       step_size)
            position = [position[0] - step_size, position[1]]
            position = self.clip_position(position)
        
        # building brick: 4 left_brick, 5 right_brick, 6 up_brick, 7 down_brick
        elif action > 3:
            build_brick = True
            if action == 4 and check[4] == 0:
                count_brick += 1
                brick_target = [position[0], position[1] - 1]
                environment_memory[brick_target[0], brick_target[1]] += 1.0
            elif action == 5 and check[5] == 0:
                count_brick += 1
                brick_target = [position[0], position[1] + 1]
                environment_memory[brick_target[0], brick_target[1]] += 1.0
            elif action == 6 and check[6] == 0:
                count_brick += 1
                brick_target = [position[0] + 1, position[1]]
                environment_memory[brick_target[0], brick_target[1]] += 1.0
            elif action == 7 and check[7] == 0:
                count_brick += 1
                brick_target = [position[0] - 1, position[1]]
                environment_memory[brick_target[0], brick_target[1]] += 1.0
            else:
                build_brick = False

            done = bool(count_brick >= self.total_brick) or check[0:4] == [1, 1, 1, 1]
            if done:
                observation = np.hstack(
                    (self.observation_transition(environment_memory,position), np.array([[count_brick]]), np.array([[count_step]])))
                reward = 0.0
                state  = position, environment_memory, count_brick, count_step
                return state, observation, reward, done
            else:
                if build_brick:
                    reward = self.reward_check_transition(environment_memory,brick_target)
                    observation = np.hstack(
                        (self.observation_transition(environment_memory,position), np.array([[count_brick]]), np.array([[count_step]])))
                    state  = position, environment_memory, count_brick, count_step
                    return state, observation, reward, done
        else:
            # illegal_move = useless_move!!
            position = position
            
        done = bool(count_step >= self.total_step) or check[0:4] == [1, 1, 1, 1]
        observation = np.hstack(
            (self.observation_transition(environment_memory,position), np.array([[count_brick]]), np.array([[count_step]])))
        reward = 0.0
        state  = position, environment_memory, count_brick, count_step
        return state, observation, reward, done

    def step(self, action):
        self.count_step += 1
        self.step_size = np.random.randint(1, 4)
        # 0 left, 1 right, 2 up, 3 down, 4 left_brick, 5 right_brick, 6 up_brick, 7 down_brick

        check = self.check_sur(self.position_memory[-1])
        self.check = check
        # moving: 0 left, 1 right, 2 up, 3 down
        if action == 0 and check[0] == 0:
            step_size = self.move_step([self.position_memory[-1][0], self.position_memory[-1][1]], action,
                                       self.step_size)
            position = [self.position_memory[-1][0], self.position_memory[-1][1] - step_size]
            position = self.clip_position(position)
            self.position_memory.append(position)
        elif action == 1 and check[1] == 0:
            step_size = self.move_step([self.position_memory[-1][0], self.position_memory[-1][1]], action,
                                       self.step_size)
            position = [self.position_memory[-1][0], self.position_memory[-1][1] + step_size]
            position = self.clip_position(position)
            self.position_memory.append(position)
        elif action == 2 and check[2] == 0:
            step_size = self.move_step([self.position_memory[-1][0], self.position_memory[-1][1]], action,
                                       self.step_size)
            position = [self.position_memory[-1][0] + step_size, self.position_memory[-1][1]]
            position = self.clip_position(position)
            self.position_memory.append(position)
        elif action == 3 and check[3] == 0:
            step_size = self.move_step([self.position_memory[-1][0], self.position_memory[-1][1]], action,
                                       self.step_size)
            position = [self.position_memory[-1][0] - step_size, self.position_memory[-1][1]]
            position = self.clip_position(position)
            self.position_memory.append(position)

        # building brick: 4 left_brick, 5 right_brick, 6 up_brick, 7 down_brick
        elif action > 3:
            build_brick = True
            position = self.position_memory[-1]
            self.position_memory.append(position)
            if action == 4 and check[4] == 0:
                self.count_brick += 1
                brick_target = [position[0], position[1] - 1]
                self.environment_memory[brick_target[0], brick_target[1]] += 1.0
            elif action == 5 and check[5] == 0:
                self.count_brick += 1
                brick_target = [position[0], position[1] + 1]
                self.environment_memory[brick_target[0], brick_target[1]] += 1.0
            elif action == 6 and check[6] == 0:
                self.count_brick += 1
                brick_target = [position[0] + 1, position[1]]
                self.environment_memory[brick_target[0], brick_target[1]] += 1.0
            elif action == 7 and check[7] == 0:
                self.count_brick += 1
                brick_target = [position[0] - 1, position[1]]
                self.environment_memory[brick_target[0], brick_target[1]] += 1.0
            else:
                build_brick = False

            done = bool(self.count_brick >= self.total_brick) or check[0:4] == [1, 1, 1, 1]
            if done:
                observation = np.hstack(
                    (self.observation_(position), np.array([[self.count_brick]]), np.array([[self.count_step]])))
                reward = 0.0

                environment_memory=self.environment_memory.copy()
                count_brick=self.count_brick
                count_step=self.count_step
                self.state = (position,environment_memory,count_brick,count_step)
                return self.state, observation, reward, done
            else:
                if build_brick:
                    reward = self.reward_check(brick_target)
                    observation = np.hstack(
                        (self.observation_(position), np.array([[self.count_brick]]), np.array([[self.count_step]])))
                    environment_memory=self.environment_memory.copy()
                    count_brick=self.count_brick
                    count_step=self.count_step
                    self.state = (position,environment_memory,count_brick,count_step)
                    return self.state, observation, reward, done
                    
        else:
            # illegal_move = useless_move!!
            position = self.position_memory[-1]

        done = bool(self.count_step >= self.total_step) or check[0:4] == [1, 1, 1, 1]
        observation = np.hstack(
            (self.observation_(position), np.array([[self.count_brick]]), np.array([[self.count_step]])))
        reward = 0.0
        environment_memory=self.environment_memory.copy()
        count_brick=self.count_brick
        count_step=self.count_step
        self.state = (position,environment_memory,count_brick,count_step)
        return self.state, observation, reward, done

    def reward_check(self, position):
        if self.environment_memory[position[0], position[1]] > self.plan[position[0], position[1]]:
            reward = -1.0
        elif self.environment_memory[position[0], position[1]] == self.plan[position[0], position[1]]:
            reward = 10.0
        else:
            reward = 1.0
        return reward
    def reward_check_transition(self,environment_memory,position):
        if environment_memory[position[0], position[1]] > self.plan[position[0], position[1]]:
            reward = -1.0
        elif environment_memory[position[0], position[1]] == self.plan[position[0], position[1]]:
            reward = 10.0
        else:
            reward = 1.0
        return reward

    def plot_3d(self, plan, Env=False):
        if Env:
            z_max = max(max(row) for row in plan)
        else:
            z_max = 1
        X, Y, Z = [], [], []
        for i in range(int(z_max)):
            for j in range(len(plan)):
                for k in range(len(plan[0])):
                    if self.environment_length > plan[j][k] > 0:
                        X.append(k)
                        Y.append(j)
                        Z.append(plan[j][k])
            plan = plan - 1.0
        return X, Y, Z

    def iou(self):

        plan = self.plan[self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_height,
               self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_width]

        env_memory = self.environment_memory[self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_height,
                     self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_width]

        env_memory_3d = np.zeros((self.plan_height, self.plan_width))
        for i in range(self.plan_height):
            for j in range(self.plan_width):
                if env_memory[i][j] > plan[i][j]:
                    env_memory_3d[i][j] = plan[i][j]
                else:
                    env_memory_3d[i][j] = env_memory[i][j]
        plan_area = self.total_brick
        env_area = self.count_brick
        cross_area = sum(map(sum, env_memory_3d))
        iou = cross_area / (plan_area + env_area - cross_area)
        return iou

        pass
    def equality_operator(self, o1, o2):
        Equal=True 
        if not np.array_equal(o1,o2):
            return False
        return Equal

    def render(self, axe, axe2, iou_min=None, iou_average=None, iter_times=100, best_env=np.array([]), best_iou=None,
               best_step=0, best_brick=0, position_memo=[0, 0]):
        axe.clear()
        axe2.clear()

        axe.set_xlim(0, self.plan_width)
        axe.set_ylim(0, self.plan_height)
        axe.set_zlim(0, self.plan_length)

        axe2.set_xlabel('X-axis')
        axe2.set_xlim(-1, self.plan_width)
        axe2.set_ylabel('Y-axis')
        axe2.set_ylim(-1, self.plan_height)

        plan = self.plan[self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_height,
               self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_width]

        plan_2d = np.zeros((self.plan_height, self.plan_width))
        IOU = self.iou()
        if best_env.any():
            env_memory = best_env[self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_height, \
                         self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_width]
            IOU = best_iou
            step = best_step
            brick = best_brick
            best_posi = position_memo
        else:
            env_memory = self.environment_memory[self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_height, \
                         self.HALF_WINDOW_SIZE:self.HALF_WINDOW_SIZE + self.plan_width]
            step = self.count_step
            brick = self.count_brick
            best_posi = self.position_memory[-1]

        env_memory_2d = np.zeros((self.plan_height, self.plan_width))
        for i in range(self.plan_height):
            for j in range(self.plan_width):
                if plan[i][j] > 0:
                    plan_2d[i][j] = 1
                if env_memory[i][j] > 0:
                    env_memory_2d[i][j] = 1

        background = np.zeros((self.plan_height, self.plan_width))
        img = np.stack((env_memory_2d, plan_2d, background), axis=2)

        # plot_plan_surface
        X, Y, Z = self.plot_3d(plan, Env=False)

        axe.scatter(X, Y, Z, marker='s', s=40, color='r', edgecolor="k", alpha=1)

        x1, y1, z1 = self.plot_3d(env_memory, Env=True)

        width = depth = 1
        height = np.zeros_like(z1)
        if x1:
            axe.bar3d(x1, y1, height, width, depth, z1, color='b', shade=True, alpha=0.15, edgecolor="k")

        axe.scatter(best_posi[1] - self.HALF_WINDOW_SIZE + 0.5,
                    best_posi[0] - self.HALF_WINDOW_SIZE + 0.5, zs=0, marker="*", color='b', s=50)

        if iou_min is None:
            iou_min = IOU
        if iou_average is None:
            iou_average = IOU
        axe.title.set_text('step=%d,used_paint=%d,IOU=%.3f' % (step, brick, IOU))
        axe2.text(0, 20, 'Iou_min_iter_%d = %.3f' % (iter_times, iou_min), color='r', fontsize=12)
        axe2.text(0, 21.2, 'Iou_average_iter_%d = %.3f' % (iter_times, iou_average), color='b', fontsize=12)

        axe2.axis('off')
        plt.imshow(img)
        axe2.plot(best_posi[1] - self.HALF_WINDOW_SIZE,
                  best_posi[0] - self.HALF_WINDOW_SIZE, '*', color='b')
        plt.draw()


if __name__ == "__main__":
    env = deep_mobile_printing_3d1r(plan_choose=0)
    fig = plt.figure(figsize=[10, 5])
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    total_reward = 0
    obs = env.reset()
    print("start!")
    env.render(ax1, ax2)
    total_iou = 0
    iou_aver = 0
    iou_min = 1
    for j in range(1):
        print('^^^^^^^^^^^^^^', j)
        start_time_test = time.time()
        env.reset()
        total_reward = 0
        print('total_brick:', env.total_brick)
        while True:
            item = [0, 1, 2, 3, 4, 5, 6, 7]
            action = np.random.choice(item, p=[0.2, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05, ])
            obs, reward, done = env.step(action)

            total_reward += reward
            env.render(ax1, ax2, iou_average=iou_aver, iou_min=iou_min, iter_times=j)
            plt.pause(0.001)
            if done:
                iou = env.iou()
                iou_min = min(iou_min, iou)
                total_iou += iou
                iou_aver = total_iou / (j + 1)
                secs = int(time.time() - start_time_test)
                mins = secs / 60
                secs = secs % 60
                env.render(ax1, ax2, iou_average=iou_aver, iou_min=iou_min, iter_times=j + 1)
                print(" | time in %d minutes, %d seconds\n" % (mins, secs))
                break
        print("total_reward", total_reward)
    env.render(ax1, ax2, iou_average=iou_aver, iou_min=iou_min, iter_times=j)

    plt.show()
