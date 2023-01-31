import time
from torch.utils.tensorboard import SummaryWriter
from arguments import get_args
from common.utils import create_log_dir, print_args, set_global_seeds, save_hyperparameters
from env.Env1D import Env1DStatic, Env1DDynamic
from env.Env2D import Env2DStatic, Env2DDynamic
from env.Env3D import Env3DStatic, Env3DDynamic
from train import train
from validate import validate
import argparse


def main():
    args = get_args()
    if args is None:
        print('Please provide a config path')
        return
    print(args)
    if args.evaluate:
        if args.env == "1DStatic":
            env = Env1DStatic(args)
        elif args.env == "1DDynamic":
            env = Env1DDynamic(args, data_path="../../Env/1D/data_1d_dynamic_sin_envplan_500_test.pkl", random_choose_paln=False)
        elif args.env == "2DStatic":
            env = Env2DStatic(args)
        elif args.env == "2DDynamic":
            if args.plan_choose == 0:
                env = Env2DDynamic(args, data_path="../../Env/2D/data_2d_dynamic_dense_envplan_500_test.pkl", random_choose_paln=False)
            else:
                env = Env2DDynamic(args, data_path="../../Env/2D/data_2d_dynamic_sparse_envplan_500_test.pkl", random_choose_paln=False)
        elif args.env == "3DStatic":
            env = Env3DStatic(args)
        elif args.env == "3DDynamic":
            if args.plan_choose == 0:
                env = Env3DDynamic(args, data_path="../../Env/3D/data_3d_dynamic_dense_envplan_500_test.pkl", random_choose_paln=False)
            else:
                env = Env3DDynamic(args, data_path="../../Env/3D/data_3d_dynamic_sparse_envplan_500_test.pkl", random_choose_paln=False)
        env.seed(args.seed)    
    else:
        if args.env == "1DStatic":
            env_train = Env1DStatic(args)
            env_val = Env1DStatic(args)
        elif args.env == "1DDynamic":
            env_train = Env1DDynamic(args, data_path="../../Env/1D/data_1d_dynamic_sin_envplan_500_train.pkl")
            env_val = Env1DDynamic(args, data_path="../../Env/1D/data_1d_dynamic_sin_envplan_500_val.pkl",random_choose_paln=False)
        elif args.env == "2DStatic":
            env_train = Env2DStatic(args)
            env_val = Env2DStatic(args)
        elif args.env == "2DDynamic":
            if args.plan_choose == 0:
                env_train = Env2DDynamic(args, data_path="../../Env/2D/data_2d_dynamic_dense_envplan_500_train.pkl")
                env_val = Env2DDynamic(args, data_path="../../Env/2D/data_2d_dynamic_dense_envplan_500_val.pkl",random_choose_paln=False)
            else:
                env_train = Env2DDynamic(args, data_path="../../Env/2D/data_2d_dynamic_sparse_envplan_500_train.pkl")
                env_val = Env2DDynamic(args, data_path="../../Env/2D/data_2d_dynamic_sparse_envplan_500_val.pkl",random_choose_paln=False)
        elif args.env == "3DStatic":
            env_train = Env3DStatic(args)
            env_val = Env3DStatic(args)
        elif args.env == "3DDynamic":
            if args.plan_choose == 0:
                env_train = Env3DDynamic(args, data_path="../../Env/3D/data_3d_dynamic_dense_envplan_500_train.pkl")
                env_val = Env3DDynamic(args, data_path="../../Env/3D/data_3d_dynamic_dense_envplan_500_val.pkl",random_choose_paln=False)
            else:
                env_train = Env3DDynamic(args, data_path="../../Env/3D/data_3d_dynamic_sparse_envplan_500_train.pkl")
                env_val = Env3DDynamic(args, data_path="../../Env/3D/data_3d_dynamic_sparse_envplan_500_val.pkl",random_choose_paln=False)

        env_train.seed(args.seed)
        env_val.seed(args.seed)

        datetime = time.time()
        save_hyperparameters(args, datetime)
        log_dir = create_log_dir(args)
        writer = SummaryWriter(log_dir)
    set_global_seeds(args.seed)
    if args.evaluate:
        validate(env, args)
    else:
        train(env_train,env_val, args, writer, datetime)
        writer.flush()
        writer.close()
        env_train.close()
        env_val.close()

if __name__ == "__main__":
    main()