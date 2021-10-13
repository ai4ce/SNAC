import matplotlib.pyplot as plt
import torch

from common.utils import save_model


def test(env, args, current_model, best_iou, writer, episode, datetime):
    total_reward = 0
    total_iou = 0
    lowest_reward = 1e4
    highest_iou = -1.0
    lowest_iou = 1.0
    plan = env.plan
    episode_reward = 0
    count_brick_save = None
    count_step_save = None

    if args.env  in ["3DStatic", "3DDynamic"]:
        fig = plt.figure(figsize=[10, 5])
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2)
    else:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

    for i in range(args.evaluation_episodes):

        state = env.reset()
        count_brick = 0
        count_step = 0

        while True:
            count_step += 1

            if args.noisy:
                current_model.sample_noise()

            epsilon = 0.0
            if args.env in [ '2DDynamic']:
                action = current_model.act(torch.FloatTensor(state[:][0]).to(args.device),
                                           torch.FloatTensor(state[:][1]).to(args.device),
                                           epsilon)
            else:
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

    avg_reward = total_reward / args.evaluation_episodes
    avg_iou = total_iou / args.evaluation_episodes

    writer.add_scalar("Average Reward/Test)", avg_reward, episode)
    writer.add_scalar("Average IOU/Test)", avg_iou, episode)

    print("\tTest Result - Average Reward: {} Lowest Reward: {} Average IOU: {}".format(
        avg_reward, lowest_reward, avg_iou))

    if avg_iou > best_iou:
        best_iou = avg_iou

        print("\n\nNEW TOP RESULT.\n\n")

        if args.env in ["3DStatic", "3DDynamic"]:
            ax1.clear()
            ax2.clear()
            env.render(ax1, ax2, args, best_env_memory, plan, highest_iou, count_step_save,
                       count_brick_save, datetime, iou_min=lowest_iou, iou_average=avg_iou)
        else:
            ax.clear()
            env.render(ax, args, best_env_memory, plan, highest_iou, count_step_save,
                       count_brick_save, datetime, iou_min=lowest_iou, iou_average=avg_iou)

        save_model(current_model, args, datetime)

    plt.close(fig)
    return best_iou

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret