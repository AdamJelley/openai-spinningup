from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=-1)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default='data/PPO/')
    args = parser.parse_args()
    eg = ExperimentGrid(name='PPO')
    eg.add('env_name', 'CartPole-v0', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 100)
    eg.add('steps_per_epoch', 2000)
    eg.add('gamma', 0.99)
    eg.add('clip_ratio', [0.1,0.3])
    eg.add('pi_lr', 0.0003)
    eg.add('vf_lr', 0.001)
    eg.add('train_pi_iters', [40,80])
    eg.add('train_v_iters', 40)
    eg.add('lam', 0.97)
    eg.add('target_kl', [0.01, 0.05])
    eg.add('ac_kwargs:hidden_sizes', (32,32), 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')
    eg.run(ppo_pytorch, num_cpu=args.cpu)