"""
UCT Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s ,a , is_model_dynamic)
env.equality_operator(s1, s2)
"""

import itertools
import mcts_Qvalue_with_iou
# =============================================================================
# import dyna_gym.utils.utils as utils
# =============================================================================
from math import sqrt, log
from gym import spaces

def uct_tree_policy(ag, children):
    return max(children, key=ag.ucb)

class UCT(object):
    """
    UCT agent
    """
    def __init__(self, action_space, rollouts=100, horizon=100, gamma=0.9, ucb_constant=6.36396103068, is_model_dynamic=True):
        if type(action_space) == spaces.discrete.Discrete:
            self.action_space = list(mcts_Qvalue_with_iou.combinations(action_space))
        else:
            self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.rollouts = rollouts
        self.horizon = horizon
        self.gamma = gamma
        self.ucb_constant = ucb_constant
        self.is_model_dynamic = is_model_dynamic

    def reset(self, p=None):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        if p == None:
            self.__init__(self.action_space)
        else:
            utils.assert_types(p,[spaces.discrete.Discrete, int, int, float, float, bool])
            self.__init__(p[0], p[1], p[2], p[3], p[4], p[5])

    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying UCT agent:')
        print('Action space       :', self.action_space)
        print('Number of actions  :', self.n_actions)
        print('Rollouts           :', self.rollouts)
        print('Horizon            :', self.horizon)
        print('Gamma              :', self.gamma)
        print('UCB constant       :', self.ucb_constant)
        print('Is model dynamic   :', self.is_model_dynamic)

    def ucb(self, node):
        """
        Upper Confidence Bound of a chance node
        """
       
        return mcts_Qvalue_with_iou.chance_node_value(node) + self.ucb_constant * sqrt(log(node.parent.visits)/len(node.sampled_returns))

    def act(self, env,Q_Net,state, obs, done,device):
        return mcts_Qvalue_with_iou.mcts_procedure(self, uct_tree_policy,Q_Net,state, obs,env, done,device)
