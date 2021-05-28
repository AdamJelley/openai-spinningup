from spinup.utils.run_utils import ExperimentGrid
from spinup import vpg_pytorch
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=-1)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()
    eg = ExperimentGrid(name='VPG')
    eg.add('env_name', 'CartPole-v0', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 100)
    eg.add('steps_per_epoch', [2000,4000])
    eg.add('gamma', 0.99)
    eg.add('pi_lr', 0.0003)
    eg.add('vf_lr', 0.001)
    eg.add('train_v_iters', [40,80])
    eg.add('ac_kwargs:hidden_sizes', [(32,), (32,32)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')
    eg.run(vpg_pytorch, num_cpu=args.cpu)