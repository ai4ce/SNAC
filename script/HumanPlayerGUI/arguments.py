import argparse

def get_args():
    parser = argparse.ArgumentParser(description='DQN')

    # Basic Arguments
    parser.add_argument('--results-path', type=str, default='results/',
                        help='Path to folder containing results.')
    parser.add_argument('--extra-note', type=str, default='dim6only',
                        help='any other changes to note about the current experiment.')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    # Training Arguments
    parser.add_argument('--max-frames', type=int, default=1500000, metavar='STEPS',
                        help='Number of frames to train')
    parser.add_argument('--buffer-size', type=int, default=10000, metavar='CAPACITY',
                        help='Maximum memory buffer size')
    parser.add_argument('--update-target', type=int, default=750, metavar='STEPS',
                        help='Interval of target network update')
    parser.add_argument('--train-freq', type=int, default=1, metavar='STEPS',
                        help='Number of steps between optimization step')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='γ',
                        help='Discount factor')
    parser.add_argument('--learning-start', type=int, default=1000, metavar='N',
                        help='How many steps of the model to collect transitions for before learning starts')
    parser.add_argument('--eps_start', type=float, default=1.0,
                        help='Start value of epsilon')
    parser.add_argument('--eps_final', type=float, default=0.01,
                        help='Final value of epsilon')
    parser.add_argument('--eps_decay', type=int, default=30000,
                        help='Adjustment parameter for epsilon')
    parser.add_argument('--half_window_size', type=int, default=2,
                        help='Half the size of partially-observable window.')
    parser.add_argument('--max-episode-length', type=int, default=int(750), metavar='LENGTH',
                        help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--history-length', type=int, default=1, metavar='T',
                        help='Number of consecutive states processed')
    parser.add_argument('--evaluation-episodes', type=int, default=20, metavar='N',
                        help='Number of evaluation episodes to average over')

    # Algorithm Arguments
    parser.add_argument('--double', action='store_true',
                        help='Enable Double-Q Learning')
    parser.add_argument('--dueling', action='store_true',
                        help='Enable Dueling Network')
    parser.add_argument('--noisy', action='store_true',
                        help='Enable Noisy Network')
    parser.add_argument('--prioritized-replay', action='store_true',
                        help='enable prioritized experience replay')
    parser.add_argument('--c51', action='store_true',
                        help='enable categorical dqn')
    parser.add_argument('--multi-step', type=int, default=3,
                        help='N-Step Learning')
    parser.add_argument('--Vmin', type=int, default=-5,
                        help='Minimum value of support for c51')
    parser.add_argument('--Vmax', type=int, default=35,
                        help='Maximum value of support for c51')
    parser.add_argument('--num-atoms', type=int, default=81,
                        help='Number of atom for c51')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha value for prioritized replay')
    parser.add_argument('--beta-start', type=float, default=0.4,
                        help='Start value of beta for prioritized replay')
    parser.add_argument('--beta-frames', type=int, default=100000,
                        help='End frame of beta schedule for prioritized replay')
    parser.add_argument('--sigma-init', type=float, default=0.1,
                        help='Sigma initialization value for NoisyNet')

    # Environment Arguments
    parser.add_argument('--env', type=str, default='1DStatic',
                        help='Environment Name')
    parser.add_argument('--episode-life', type=int, default=1,
                        help='Whether env has episode life(1) or not(0)')
    parser.add_argument('--clip-rewards', type=int, default=0,
                        help='Whether env clip rewards(1) or not(0)')
    parser.add_argument('--frame-stack', type=int, default=0,
                        help='Whether env stacks frame(1) or not(0)')
    parser.add_argument('--scale', type=int, default=0,
                        help='Whether env scales(1) or not(0)')
    parser.add_argument('--uniform_step', action='store_true',
                        help='Enable Double-Q Learning')
    parser.add_argument('--plan-choose', type=int, default=None,
                        help='Specifies which plan to train/test on.')

    # Evaluation Arguments
    parser.add_argument('--load-model', type=str, default=None,
                        help='Pretrained model name to load (state dict)')
    parser.add_argument('--save-model', type=str, default='model',
                        help='Pretrained model name to save (state dict)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate only')
    parser.add_argument('--render', action='store_true',
                        help='Render evaluation agent')
    parser.add_argument('--evaluation-interval', type=int, default=750,
                        help='Frames for evaluation interval')

    # Optimization Arguments
    parser.add_argument('--lr', type=float, default=.00005, metavar='η',
                        help='Learning rate')

    args = parser.parse_args()

    return args
