# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 19:36:49 2021

@author: lelea
"""

"""
MCTS Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s ,a , is_model_dynamic)
env.equality_operator(s1, s2)
"""

import random
import itertools
# =============================================================================
# import dyna_gym.utils.utils as utils
# =============================================================================
from gym import spaces
from math import sqrt, log
from copy import copy
import torch
import numpy as np 
def chance_node_value(node):
    """
    Value of a chance node
    """
    return sum(node.sampled_returns) / len(node.sampled_returns)

def decision_node_value(node):
    """
    Value of a decision node
    """
    return chance_node_value(max(node.children, key=chance_node_value))

def combinations(space):
    if isinstance(space, spaces.Discrete):
        return range(space.n)
    elif isinstance(space, spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError

def mcts_tree_policy(ag, children):
    return random.choice(children)

def mcts_procedure(ag, tree_policy,Q_net, state, obs, env, done, device):
    """
    Compute the entire MCTS procedure wrt to the selected tree policy.
    Funciton tree_policy is a function taking an agent + a list of ChanceNodes as argument
    and returning the one chosen by the tree policy.
    """
    root = DecisionNode(None,state, obs, ag.action_space.copy(), done)
    env_plan=torch.from_numpy(env.plan).float().to(device)
    env_plan=env_plan.unsqueeze(0)
# =============================================================================
#     print(env.state)
# =============================================================================
    # print(root.state)
    for mmm in range(ag.rollouts):
        rewards = [] # Rewards collected along the tree for the current rollout
        node = root # Current node
        terminal = done
        # print("*************************************************************")
        # print("number of rollout",mmm)
        # print(node.state)
        # print(node.obs)

        # Selection
        select = True
        while select:
            if (type(node) == DecisionNode): # DecisionNode
                if node.is_terminal:
                    select = False # Selected a terminal DecisionNode
                else:
                    if len(node.children) < ag.n_actions:
                        select = False # Selected a non-fully-expanded DecisionNode
                    else:
                        #node = random.choice(node.children) #TODO remove
                        node = tree_policy(ag, node.children)
            else: # ChanceNode
                # print("ChanceNode")
                # print(node.parent.state)
                # print(node.action)
                state_p, obs_p, reward, terminal = env.transition(node.parent.state, node.action, ag.is_model_dynamic)
                # print(state_p)
                # print(obs_p)
                Q_value=Q_net(torch.FloatTensor(node.parent.obs).to(device),torch.FloatTensor(np.array([node.action])).unsqueeze(0).to(device),env_plan)
                rewards.append(Q_value)
                # print("ChanceNode's children",node.children)

                if (len(node.children) == 0):
                    select = False # Selected a ChanceNode
                else:
                    new_state = True
                    for i in range(len(node.children)):
                       
                        if env.equality_operator(node.children[i].obs, obs_p):
                            node = node.children[i]
                            new_state = False
                           
                            break
                    if new_state:
                        select = False # Selected a ChanceNode

        # Expansion
        if (type(node) == ChanceNode) or ((type(node) == DecisionNode) and not node.is_terminal):
            if (type(node) == DecisionNode):
                node.children.append(ChanceNode(node, node.possible_actions.pop()))
                # print(node.children)
                node = node.children[-1]
                # print(node.parent.state)
                # print(node.action)
                state_p, obs_p, reward, terminal = env.transition(node.parent.state ,node.action, ag.is_model_dynamic)
                # print(state_p)
                # print(obs_p)
                Q_value=Q_net(torch.FloatTensor(node.parent.obs).to(device),torch.FloatTensor(np.array([node.action])).unsqueeze(0).to(device),env_plan)
                rewards.append(Q_value)
            # ChanceNode
            node.children.append(DecisionNode(node, state_p, obs_p, ag.action_space.copy(), terminal))
           
            node = node.children[-1]

        # Evaluation
        assert(type(node) == DecisionNode)
        estimate = Q_value
        # print(estimate)
        # print(rewards)
        # Backpropagation
        node.visits += 1
        node = node.parent
        assert(type(node) == ChanceNode)
        while node:
            node.sampled_returns.append(estimate)
            if len(rewards) != 0:
                estimate = rewards.pop() + ag.gamma * estimate
            node.parent.visits += 1
            node = node.parent.parent
    # print(root.state)
    # print(root.obs)
    # print(root.children)
    return max(root.children, key=chance_node_value).action

class DecisionNode:
    """
    Decision node class, labelled by a state
    """
    def __init__(self, parent, state, obs, possible_actions, is_terminal):
        self.parent = parent
        self.state = state
        self.obs=obs
        self.is_terminal = is_terminal
        if self.parent is None: # Root node
            self.depth = 0
        else: # Non root node
            self.depth = parent.depth + 1
        self.children = []
        self.possible_actions = possible_actions
        random.shuffle(self.possible_actions)
        self.explored_children = 0
        self.visits = 0

class ChanceNode:
    """
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    """
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.depth = parent.depth
        self.children = []
        self.sampled_returns = []
