import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from common.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from common.utils import epsilon_scheduler, beta_scheduler, update_target, print_log, load_model
from models.model_1D import DQN_1D
from models.model_2D import DQN_2D
from models.model_3D import DQN_3D
from test import test


def train(env, args, writer, datetime):
    best_iou = -1.0
    if args.env in ['1DStatic', '1DDynamic']:
        current_model = DQN_1D(env, args).to(args.device)
        target_model = DQN_1D(env, args).to(args.device)
    elif args.env in ['2DStatic', '2DDynamic']:
        current_model = DQN_2D(env, args).to(args.device)
        target_model = DQN_2D(env, args).to(args.device)
    elif args.env in ['3DStatic', '3DDynamic']:
        current_model = DQN_3D(env, args).to(args.device)
        target_model = DQN_3D(env, args).to(args.device)

    if args.noisy:
        current_model.update_noisy_modules()
        target_model.update_noisy_modules()

    if args.load_model and os.path.isfile(args.load_model):
        load_model(current_model, args)

    epsilon_by_frame = epsilon_scheduler(args.eps_start, args.eps_final, args.eps_decay)
    beta_by_frame = beta_scheduler(args.beta_start, args.beta_frames)

    if args.prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(args.buffer_size, args.alpha)
    else:
        replay_buffer = ReplayBuffer(args.buffer_size)
    
    state_deque = deque(maxlen=args.multi_step)
    reward_deque = deque(maxlen=args.multi_step)
    action_deque = deque(maxlen=args.multi_step)

    optimizer = optim.Adam(current_model.parameters(), lr=args.lr)

    reward_list, length_list, loss_list = [], [], []
    episode_reward = 0
    episode_length = 0
    episode = 0

    prev_time = time.time()
    prev_frame = 1

    state = env.reset()
    for frame_idx in range(1, args.max_frames + 1):
        if args.render:
            env.render()

        if args.noisy:
            current_model.sample_noise()
            target_model.sample_noise()

        epsilon = epsilon_by_frame(frame_idx)
        if args.env in ['2DDynamic']:
            action = current_model.act(torch.FloatTensor(state[:][0]).to(args.device),
                                       torch.FloatTensor(state[:][1]).to(args.device),
                                       epsilon)
        else:
            action = current_model.act(torch.FloatTensor(state).to(args.device), epsilon)
        next_state, reward, done = env.step(action)
        state_deque.append(state)
        reward_deque.append(reward)
        action_deque.append(action)

        if len(state_deque) == args.multi_step or done:
            n_reward = multi_step_reward(reward_deque, args.gamma)
            n_state = state_deque[0]
            n_action = action_deque[0]
            replay_buffer.push(n_state, n_action, n_reward, next_state, np.float32(done))

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done:
            episode += 1

            state = env.reset()
            reward_list.append(episode_reward)
            length_list.append(episode_length)
            writer.add_scalar("Episode_reward/train", episode_reward, episode)
            writer.add_scalar("Episode_length/train", episode_length, episode)
            episode_reward = 0
            episode_length = 0
            state_deque.clear()
            reward_deque.clear()
            action_deque.clear()

        if len(replay_buffer) > args.learning_start and frame_idx % args.train_freq == 0:
            beta = beta_by_frame(frame_idx)
            loss = compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta)
            loss_list.append(loss.item())
            writer.add_scalar("Loss/train", loss.item(), frame_idx)

        if frame_idx % args.update_target == 0:
            update_target(current_model, target_model)

        if frame_idx % args.evaluation_interval == 0:
            print_log(frame_idx, prev_frame, prev_time, reward_list, length_list, loss_list, args)
            reward_list.clear(), length_list.clear(), loss_list.clear()
            prev_frame = frame_idx
            prev_time = time.time()

            best_iou = test(env, args, current_model, best_iou, writer, episode, datetime)

def compute_td_loss(current_model, target_model, replay_buffer, optimizer, args, beta=None):
    """
    Calculate loss and optimize for non-c51 algorithm
    """
    if args.prioritized_replay:
        state, action, reward, next_state, done, weights, indices = replay_buffer.sample(args.batch_size, beta)
    else:
        state, action, reward, next_state, done = replay_buffer.sample(args.batch_size)
        weights = torch.ones(args.batch_size)

    if args.env in ['1DStatic', '2DStatic', '3DStatic']:
        state = torch.FloatTensor(np.float32(state)).to(args.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(args.device)
    action = torch.LongTensor(action).to(args.device)
    reward = torch.FloatTensor(reward).to(args.device)
    done = torch.FloatTensor(done).to(args.device)
    weights = torch.FloatTensor(weights).to(args.device)

    if not args.c51:

        if args.env in ['2DDynamic']:
            q_values = current_model(torch.FloatTensor(np.float32(state[:][0])).to(args.device),
                                     torch.FloatTensor(np.float32(state[:][1])).to(args.device))
        elif args.env in ['1DDynamic','3DDynamic']:
            q_values = current_model(torch.FloatTensor(state).to(args.device))
        else:
            q_values = current_model(state)
        target_next_q_values = target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        if args.double:
            if args.env in ['2DDynamic']:
                next_q_values = current_model(torch.FloatTensor(np.float32(next_state[:][0])).to(args.device),
                                         torch.FloatTensor(np.float32(next_state[:][1])).to(args.device))
            elif args.env in  ['1DDynamic','3DDynamic']:
                next_q_values = current_model(torch.FloatTensor(next_state).to(args.device))
            else:
                next_q_values = current_model(next_state)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            next_q_value = target_next_q_values.gather(1, next_actions).squeeze(1)
        else:
            next_q_value = target_next_q_values.max(1)[0]

        expected_q_value = reward + (args.gamma ** args.multi_step) * next_q_value * (1 - done)

        loss = F.smooth_l1_loss(q_value, expected_q_value.detach(), reduction='none')
        if args.prioritized_replay:
            prios = torch.abs(loss) + 1e-5
        loss = (loss * weights).mean()
    
    else:
        if args.env in ['2DDynamic']:
            q_dist = current_model(torch.FloatTensor(np.float32(state[:][0])).to(args.device),
                                   torch.FloatTensor(np.float32(state[:][1])).to(args.device))
        elif args.env in  ['1DDynamic','3DDynamic']:
            q_dist = current_model(torch.FloatTensor(state).to(args.device))
        else:
            q_dist = current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(args.batch_size, 1, args.num_atoms)
        q_dist = q_dist.gather(1, action).squeeze(1)
        q_dist.data.clamp_(0.001, 0.999)
        target_dist = projection_distribution(current_model, target_model, next_state, reward, done, 
                                              target_model.support, target_model.offset, args)
        loss = - (target_dist * q_dist.log()).sum(1)
        if args.prioritized_replay:
            prios = torch.abs(loss) + 1e-6
        loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    if args.prioritized_replay:
        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    optimizer.step()

    return loss


def projection_distribution(current_model, target_model, next_state, reward, done, support, offset, args):
    delta_z = float(args.Vmax - args.Vmin) / (args.num_atoms - 1)
    if args.env in ['2DDynamic']:
        target_next_q_dist = target_model(torch.FloatTensor(np.float32(next_state[:][0])).to(args.device),
                                        torch.FloatTensor(np.float32(next_state[:][1])).to(args.device))
    elif args.env in  ['1DDynamic','3DDynamic']:
        target_next_q_dist = target_model(torch.FloatTensor(next_state).to(args.device))
    else:
        target_next_q_dist = target_model(next_state)
    if args.double:
        if args.env in ['2DDynamic']:
            next_q_dist = current_model(torch.FloatTensor(np.float32(next_state[:][0])).to(args.device),
                                        torch.FloatTensor(np.float32(next_state[:][1])).to(args.device))
        elif args.env in  ['1DDynamic','3DDynamic']:
            next_q_dist = current_model(torch.FloatTensor(next_state).to(args.device))
        else:
            next_q_dist = current_model(next_state)
        next_action = (next_q_dist * support).sum(2).max(1)[1]
    else:
        next_action = (target_next_q_dist * support).sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(target_next_q_dist.size(0), 1, target_next_q_dist.size(2))
    target_next_q_dist = target_next_q_dist.gather(1, next_action).squeeze(1)

    reward = reward.unsqueeze(1).expand_as(target_next_q_dist)
    done = done.unsqueeze(1).expand_as(target_next_q_dist)
    support = support.unsqueeze(0).expand_as(target_next_q_dist)
    Tz = reward + args.gamma * support * (1 - done)
    Tz = Tz.clamp(min=args.Vmin, max=args.Vmax)
    b = (Tz - args.Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()
    target_dist = target_next_q_dist.clone().zero_()
    target_dist.view(-1).index_add_(0, (l + offset).view(-1), (target_next_q_dist * (u.float() - b)).view(-1))
    target_dist.view(-1).index_add_(0, (u + offset).view(-1), (target_next_q_dist * (b - l.float())).view(-1))

    return target_dist

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret