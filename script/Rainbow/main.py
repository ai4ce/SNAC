import time
from torch.utils.tensorboard import SummaryWriter
from arguments import get_args
from common.utils import create_log_dir, print_args, set_global_seeds, save_hyperparameters
from env.Env1D import Env1DStatic, Env1DDynamic, Env1DDynamic_Validation
from env.Env2D import Env2DStatic, Env2DDynamic, Env2DDynamic_Validation
from env.Env3D import Env3DStatic, Env3DDynamic, Env3DDynamic_Validation
from train import train
from validate import validate


def main():
    args = get_args()
    print_args(args)

    if args.evaluate:
        if args.env == "1DStatic":
            env = Env1DStatic(args)
        elif args.env == "1DDynamic":
            env = Env1DDynamic_Validation(args)
        elif args.env == "2DStatic":
            env = Env2DStatic(args)
        elif args.env == "2DDynamic":
            env = Env2DDynamic_Validation(args)
        elif args.env == "3DStatic":
            env = Env3DStatic(args)
        elif args.env == "3DDynamic":
            env = Env3DDynamic_Validation(args)
    else:
        if args.env == "1DStatic":
            env = Env1DStatic(args)
        elif args.env == "1DDynamic":
            env = Env1DDynamic(args)
        elif args.env == "2DStatic":
            env = Env2DStatic(args)
        elif args.env == "2DDynamic":
            env = Env2DDynamic(args)
        elif args.env == "3DStatic":
            env = Env3DStatic(args)
        elif args.env == "3DDynamic":
            env = Env3DDynamic(args)

        datetime = time.time()
        save_hyperparameters(args, datetime)
        log_dir = create_log_dir(args)
        writer = SummaryWriter(log_dir)

    set_global_seeds(args.seed)
    env.seed(args.seed)

    if args.evaluate:
        validate(env, args)
    else:
        train(env, args, writer, datetime)
        writer.flush()
        writer.close()

    env.close()


if __name__ == "__main__":
    main()
