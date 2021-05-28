from spinup.utils.run_utils import ExperimentGrid
from spinup import trpo_tf1
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=-1)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default='data/TRPO')
    args = parser.parse_args()
    eg = ExperimentGrid(name='TRPO')
    eg.add('env_name', 'CartPole-v1', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 100)
    eg.add('steps_per_epoch', 2000)
    eg.add('gamma', 0.99)
    eg.add('delta', [0.001,0.01])
    eg.add('vf_lr', 0.001)
    eg.add('train_v_iters', 40)
    eg.add('damping_coeff', 0.1)
    eg.add('cg_iters', 10)
    eg.add('backtrack_iters', 10)
    eg.add('backtrack_coeff', 0.8)
    eg.add('lam', 0.97)
    eg.add('max_ep_len', 1000)
    eg.add('ac_kwargs:hidden_sizes', (32,32), 'hid')
    eg.add('ac_kwargs:activation', [tf.nn.relu, tf.nn.tanh], '')
    eg.add('algo', ['npg','trpo'])
    eg.run(trpo_tf1, num_cpu=args.cpu)