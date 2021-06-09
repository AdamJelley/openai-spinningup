from spinup.utils.run_utils import ExperimentGrid
from spinup import td3_pytorch
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=-1)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default='data/TD3/')
    args = parser.parse_args()
    eg = ExperimentGrid(name='TD3')
    eg.add('env_name', 'LunarLanderContinuous-v2', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 100)
    eg.add('steps_per_epoch', 4000)
    eg.add('replay_size', 1000000)
    eg.add('gamma', 0.99)
    eg.add('polyak', 0.995)
    eg.add('pi_lr', 0.001)
    eg.add('q_lr', 0.001)
    eg.add('batch_size', 100)
    eg.add('start_steps', 10000)
    eg.add('update_after', 1000)
    eg.add('update_every', 50)
    eg.add('act_noise', 0.1)
    eg.add('target_noise', 0.2)
    eg.add('noise_clip', 0.5)
    eg.add('policy_delay', 2)
    eg.add('num_test_episodes', 10)
    eg.add('max_ep_len', 1000)
    #eg.add('ac_kwargs:hidden_sizes', (32,32), 'hid')
    #eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')
    eg.run(td3_pytorch, num_cpu=args.cpu)