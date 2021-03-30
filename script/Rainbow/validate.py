import matplotlib.pyplot as plt
import torch

from common.utils import load_model
from models.model_1D import DQN_1D
from models.model_2D import DQN_2D
from models.model_3D import DQN_3D

def validate(env, args):
    if args.env == "1DStatic":
        validate_1DStatic(env, args)
    elif args.env == "1DDynamic":
        validate_1DDynamic(env, args)
    elif args.env == "2DStatic":
        validate_2DStatic(env, args)
    elif args.env == "2DDynamic":
        validate_2DDynamic(env, args)
    elif args.env == "3DStatic":
        validate_3DStatic(env, args)
    elif args.env == "3DDynamic":
        validate_3DDynamic(env, args)
    return

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret

def validate_1DStatic(env, args):
    current_model = DQN_1D(env, args).to(args.device)
    current_model.update_noisy_modules()
    load_model(current_model, args)
    current_model.eval()

    NUM_TEST_EPISODES = 500

    lowest_reward = 1e4
    highest_iou = 0.0
    lowest_iou = 1.0
    plan = env.plan
    episode_reward = 0
    count_brick_save = None
    count_step_save = None

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    test_episode_count = 0
    total_reward = 0
    total_iou = 0

    for i in range(NUM_TEST_EPISODES):
        test_episode_count += 1
        state = env.reset()
        count_brick = 0
        count_step = 0

        while True:
            count_step += 1
            if args.noisy:
                current_model.sample_noise()

            epsilon = 0.0
            with torch.no_grad():
                action = current_model.act(torch.FloatTensor(state).to(args.device), epsilon)
            if action == 2:
                count_brick += 1

            next_state, reward, done = env.step(action)

            state = next_state
            episode_reward += reward

            if done:
                total_reward += episode_reward
                lowest_reward = min(lowest_reward, episode_reward)

                environment_memory = env.environment_memory[0, args.half_window_size: 34 - args.half_window_size]
                iou = env._iou()
                total_iou += iou
                if iou > highest_iou:
                    highest_iou = iou
                    best_env_memory = environment_memory
                    count_brick_save = count_brick
                    count_step_save = count_step

                lowest_iou = min(iou, lowest_iou)

                episode_reward = 0
                break

        if test_episode_count % 5 == 0:
            print("\tTest Episode: ", test_episode_count, " / {} Average Reward: {} Average IOU: {}"
                  .format(NUM_TEST_EPISODES, total_reward / test_episode_count, total_iou / test_episode_count))

    avg_reward = total_reward / NUM_TEST_EPISODES
    avg_iou = total_iou / NUM_TEST_EPISODES

    print("\tTest Result - Average Reward: {} Lowest Reward: {} Average IOU: {}".format(
        avg_reward, lowest_reward, avg_iou))

    env.render(ax, args, best_env_memory, plan, highest_iou, count_step_save,
               count_brick_save, args.load_model, iou_min=lowest_iou, iou_average=avg_iou, iter_times=NUM_TEST_EPISODES)
    plt.close(fig)

def validate_1DDynamic(env, args):
    current_model = DQN_1D(env, args).to(args.device)
    load_model(current_model, args)
    current_model.eval()

    TEST_EPISODES_PER_PLAN = 200
    NUM_TEST_PLANS = 10

    lowest_reward = 1e4
    highest_iou = 0.0
    lowest_iou = 1.0
    episode_reward = 0
    count_brick_save = None
    count_step_save = None

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    cumulative_reward = 0
    cumulative_iou = 0

    for plan_choose in range(8, 9):
        print("Validation Plan: ", plan_choose)
        env.set_plan_choose(plan_choose)
        test_episode_count = 0
        total_reward = 0
        total_iou = 0
        for i in range(TEST_EPISODES_PER_PLAN):
            test_episode_count += 1
            state = env.reset()
            plan = env.plan
            count_brick = 0
            count_step = 0

            while True:
                count_step += 1

                epsilon = 0.0
                with torch.no_grad():
                    action = current_model.act(torch.FloatTensor(state).to(args.device), epsilon)
                if action == 2:
                    count_brick += 1

                next_state, reward, done = env.step(action)

                state = next_state
                episode_reward += reward

                if done:
                    total_reward += episode_reward
                    lowest_reward = min(lowest_reward, episode_reward)

                    environment_memory = env.environment_memory[0, args.half_window_size: 34 - args.half_window_size]
                    iou = env._iou()
                    total_iou += iou
                    if iou > highest_iou:
                        highest_iou = iou
                        best_env_memory = environment_memory
                        count_brick_save = count_brick
                        count_step_save = count_step
                    if iou < lowest_iou:
                        lowest_iou = iou

                    episode_reward = 0
                    break
            if test_episode_count % 5 == 0:
                print("\tTest Episode: ", test_episode_count, " / {} Average Reward: {} Average IOU: {}"
                      .format(TEST_EPISODES_PER_PLAN, total_reward / test_episode_count, total_iou / test_episode_count))

        avg_reward = total_reward / TEST_EPISODES_PER_PLAN
        avg_iou = total_iou / TEST_EPISODES_PER_PLAN

        cumulative_reward += avg_reward
        cumulative_iou += avg_iou

        print("\tTest Result - Plan: {} Average Reward: {} Lowest Reward: {} Average IOU: {}".format(
            plan_choose, avg_reward, lowest_reward, avg_iou))


    avg_reward_allplans = cumulative_reward / NUM_TEST_PLANS
    avg_iou_allplans = cumulative_iou / NUM_TEST_PLANS
    print("\n\tTest Result (over all plans) - Average Reward: {} Lowest Reward: {} Average IOU: {}\n\n".format(
        avg_reward_allplans, lowest_reward, avg_iou_allplans))

    env.render(ax, args, best_env_memory, plan, highest_iou, count_step_save,
               count_brick_save, args.load_model, iou_min=lowest_iou, iou_average=avg_iou_allplans,
               iter_times=TEST_EPISODES_PER_PLAN)

    plt.close(fig)


def validate_2DStatic(env, args):
    current_model = DQN_2D(env, args).to(args.device)
    load_model(current_model, args)
    current_model.eval()

    NUM_TEST_EPISODES = 500

    lowest_reward = 1e4
    highest_iou = 0.0
    lowest_iou = 1.0
    plan = env.plan
    episode_reward = 0
    count_brick_save = None
    count_step_save = None

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    test_episode_count = 0
    total_reward = 0
    total_iou = 0

    for i in range(NUM_TEST_EPISODES):
        test_episode_count += 1
        state = env.reset()
        count_brick = 0
        count_step = 0

        while True:
            count_step += 1

            epsilon = 0.0
            with torch.no_grad():
                action = current_model.act(torch.FloatTensor(state).to(args.device), epsilon)
            if action == 2:
                count_brick += 1

            next_state, reward, done = env.step(action)

            state = next_state
            episode_reward += reward

            if done:
                total_reward += episode_reward
                lowest_reward = min(lowest_reward, episode_reward)

                environment_memory = env.environment_memory[0, args.half_window_size: 34 - args.half_window_size]
                iou = env._iou()
                total_iou += iou
                if iou > highest_iou:
                    highest_iou = iou
                    best_env_memory = environment_memory
                    count_brick_save = count_brick
                    count_step_save = count_step
                if iou < lowest_iou:
                    lowest_iou = iou

                episode_reward = 0
                break

        if test_episode_count % 5 == 0:
            print("\tTest Episode: ", test_episode_count, " / {} Average Reward: {} Average IOU: {}"
                  .format(NUM_TEST_EPISODES, total_reward / test_episode_count, total_iou / test_episode_count))

    avg_reward = total_reward / NUM_TEST_EPISODES
    avg_iou = total_iou / NUM_TEST_EPISODES

    print("\tTest Result - Average Reward: {} Lowest Reward: {} Average IOU: {}".format(
        avg_reward, lowest_reward, avg_iou))

    env.render(ax, args, best_env_memory, plan, highest_iou, count_step_save,
               count_brick_save, args.load_model, iou_min=lowest_iou, iou_average=avg_iou, iter_times=NUM_TEST_EPISODES)
    plt.close(fig)

def validate_2DDynamic(env, args):
    current_model = DQN_2D(env, args).to(args.device)
    load_model(current_model, args)
    current_model.eval()

    TEST_EPISODES_PER_PLAN = 200
    NUM_TEST_PLANS = 1

    lowest_reward = 1e4
    highest_iou = 0.0
    lowest_iou = 1.0
    episode_reward = 0
    count_brick_save = None
    count_step_save = None

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    cumulative_reward = 0
    cumulative_iou = 0

    for tests_set in range(6, 7):
        print("Validation Plan: ", tests_set)
        env.set_tests_set(tests_set)
        test_episode_count = 0
        total_reward = 0
        total_iou = 0
        for i in range(TEST_EPISODES_PER_PLAN):
            test_episode_count += 1
            state = env.reset()
            plan = env.plan
            count_brick = 0
            count_step = 0

            while True:
                count_step += 1

                epsilon = 0.0
                with torch.no_grad():
                    # print(state)
                    action = current_model.act(torch.FloatTensor(state[0]).to(args.device), epsilon)
                if action == 2:
                    count_brick += 1

                next_state, reward, done = env.step(action)

                state = next_state
                episode_reward += reward

                if done:
                    total_reward += episode_reward
                    lowest_reward = min(lowest_reward, episode_reward)

                    environment_memory = env.environment_memory[0, args.half_window_size: 34 - args.half_window_size]
                    iou = env._iou()
                    print(iou)
                    total_iou += iou
                    if iou > highest_iou:
                        highest_iou = iou
                        # print("NEW HIGH: ", highest_iou)
                        best_env_memory = environment_memory
                        count_brick_save = count_brick
                        count_step_save = count_step
                    if iou < lowest_iou:
                        lowest_iou = iou
                        # print("NEW LOW: ", lowest_iou)

                    episode_reward = 0
                    break
            if test_episode_count % 5 == 0:
                print("\tTest Episode: ", test_episode_count, " / {} Average Reward: {} Average IOU: {}"
                      .format(TEST_EPISODES_PER_PLAN, total_reward / test_episode_count, total_iou / test_episode_count))

        # print(total_iou)
        avg_reward = total_reward / TEST_EPISODES_PER_PLAN
        avg_iou = total_iou / TEST_EPISODES_PER_PLAN

        cumulative_reward += avg_reward
        cumulative_iou += avg_iou

        print("\tTest Result - Plan: {} Average Reward: {} Lowest Reward: {} Average IOU: {}".format(
            tests_set, avg_reward, lowest_reward, avg_iou))

    avg_reward_allplans = cumulative_reward / NUM_TEST_PLANS
    avg_iou_allplans = cumulative_iou / NUM_TEST_PLANS
    print("\n\tTest Result (over all plans) - Average Reward: {} Lowest Reward: {} Average IOU: {}\n\n".format(
        avg_reward_allplans, lowest_reward, avg_iou_allplans))

    env.render(ax, args, best_env_memory, plan, highest_iou, count_step_save,
               count_brick_save, args.load_model, iou_min=lowest_iou, iou_average=avg_iou_allplans,
               iter_times=TEST_EPISODES_PER_PLAN)

    plt.close(fig)

def validate_3DStatic(env, args):
    current_model = DQN_3D(env, args).to(args.device)
    current_model.update_noisy_modules()
    load_model(current_model, args)
    current_model.eval()

    NUM_TEST_EPISODES = 500

    lowest_reward = 1e4
    highest_iou = 0.0
    lowest_iou = 1.0
    plan = env.plan
    episode_reward = 0
    count_brick_save = None
    count_step_save = None
    best_env_memory = None

    fig = plt.figure(figsize=[10, 5])
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    test_episode_count = 0
    total_reward = 0
    total_iou = 0

    for i in range(NUM_TEST_EPISODES):
        test_episode_count += 1
        state = env.reset()
        count_brick = 0
        count_step = 0

        while True:
            if args.noisy:
                # current_model.reset_parameters()
                current_model.sample_noise()

            count_step += 1

            epsilon = 0.0
            with torch.no_grad():
                action = current_model.act(torch.FloatTensor(state).to(args.device), epsilon)
            if action == 2:
                count_brick += 1

            next_state, reward, done = env.step(action)

            state = next_state
            episode_reward += reward

            if done:
                total_reward += episode_reward
                lowest_reward = min(lowest_reward, episode_reward)

                environment_memory = env.environment_memory[0, args.half_window_size: 34 - args.half_window_size]
                iou = env._iou()
                total_iou += iou
                if iou > highest_iou:
                    highest_iou = iou
                    best_env_memory = environment_memory
                    count_brick_save = count_brick
                    count_step_save = count_step
                if iou < lowest_iou:
                    lowest_iou = iou

                episode_reward = 0
                break

        if test_episode_count % 5 == 0:
            print("\tTest Episode: ", test_episode_count, " / {} Average Reward: {} Average IOU: {}"
                  .format(NUM_TEST_EPISODES, total_reward / test_episode_count, total_iou / test_episode_count))

    avg_reward = total_reward / NUM_TEST_EPISODES
    avg_iou = total_iou / NUM_TEST_EPISODES

    print("\tTest Result - Average Reward: {} Lowest Reward: {} Average IOU: {}".format(
        avg_reward, lowest_reward, avg_iou))

    env.render(ax1, ax2, args, best_env_memory, plan, highest_iou, count_step_save,
               count_brick_save, args.load_model, iou_min=lowest_iou, iou_average=avg_iou, iter_times=NUM_TEST_EPISODES)

    plt.close(fig)

def validate_3DDynamic(env, args):
    current_model = DQN_3D(env, args).to(args.device)
    load_model(current_model, args)
    current_model.update_noisy_modules()
    current_model.eval()

    TEST_EPISODES_PER_PLAN = 200
    NUM_TEST_PLANS = 10

    lowest_reward = 1e4
    highest_iou = 0.0
    lowest_iou = 1.0
    plan = env.plan
    episode_reward = 0
    count_brick_save = None
    count_step_save = None

    fig = plt.figure(figsize=[10, 5])
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    cumulative_reward = 0
    cumulative_iou = 0
    best_env_memory = None

    for tests_set in range(3, 4):
        print("Validation Plan: ", tests_set)
        env.set_tests_set(tests_set)
        test_episode_count = 0
        total_reward = 0
        total_iou = 0
        for i in range(TEST_EPISODES_PER_PLAN):
            test_episode_count += 1
            state = env.reset()
            count_brick = 0
            count_step = 0
            while True:
                count_step += 1

                if args.noisy:
                    current_model.reset_parameters()
                    current_model.sample_noise()

                epsilon = 0.0
                with torch.no_grad():
                    action = current_model.act(torch.FloatTensor(state).to(args.device), epsilon)
                if action == 2:
                    count_brick += 1

                next_state, reward, done = env.step(action)

                state = next_state
                episode_reward += reward

                if done:
                    total_reward += episode_reward
                    lowest_reward = min(lowest_reward, episode_reward)

                    environment_memory = env.environment_memory[0, args.half_window_size: 34 - args.half_window_size]
                    iou = env._iou()
                    total_iou += iou
                    if iou > highest_iou:
                        highest_iou = iou
                        best_env_memory = environment_memory
                        count_brick_save = count_brick
                        count_step_save = count_step

                    if iou < lowest_iou:
                        lowest_iou = iou

                    episode_reward = 0
                    break
            print("\tTest Episode: ", test_episode_count, " / {} Average Reward: {} Average IOU: {}"
                  .format(TEST_EPISODES_PER_PLAN, total_reward / test_episode_count, total_iou / test_episode_count))

        avg_reward = total_reward / TEST_EPISODES_PER_PLAN
        avg_iou = total_iou / TEST_EPISODES_PER_PLAN

        cumulative_reward += avg_reward
        cumulative_iou += avg_iou

        print("\tTest Result - Plan: {} Average Reward: {} Lowest Reward: {} Average IOU: {}".format(
            tests_set, avg_reward, lowest_reward, avg_iou))

    avg_reward_allplans = cumulative_reward / NUM_TEST_PLANS
    avg_iou_allplans = cumulative_iou / NUM_TEST_PLANS
    print("\n\tTest Result (over all plans) - Average Reward: {} Lowest Reward: {} Average IOU: {}\n\n".format(
        avg_reward_allplans, lowest_reward, avg_iou_allplans))

    env.render(ax1, ax2, args, best_env_memory, plan, highest_iou, count_step_save,
               count_brick_save, args.load_model, iou_min=lowest_iou, iou_average=avg_iou_allplans,
               iter_times=TEST_EPISODES_PER_PLAN)

    plt.close(fig)