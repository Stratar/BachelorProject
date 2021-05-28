import os
import sys
import tensorflow as tf
import gym
import numpy as np
import baselines.common.tf_util as u

from mpi4py import MPI
from osim.env import ProstheticsEnvMulticlip
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import MultiCategoricalPdType
from baselines.ppo1 import pposgd_simple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = MultiCategoricalPdType(low=np.zeros_like(ac_space.low, dtype=np.int32),
                                                      high=np.ones_like(ac_space.high, dtype=np.int32))
        gaussian_fixed_var = True

        sequence_length = None

        ob = u.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i" % (i + 1),
                                                      kernel_initializer=u.normc_initializer(1.0)))  # tanh
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=u.normc_initializer(1.0))[:, 0]

        with tf.variable_scope('pol'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i' % (i + 1),
                                                      kernel_initializer=u.normc_initializer(1.0)))  # tanh
            
            #print(last_out)
            self.pdparam = pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final_pol',
                                      kernel_initializer=u.normc_initializer(0.01))

            self.pi_vpred = tf.layers.dense(last_out, 1, name='final_val', kernel_initializer=u.normc_initializer(1.0))[:, 0]
        low = np.zeros_like(ac_space.low, dtype=np.int32)
        high = np.ones_like(ac_space.high, dtype=np.int32)
        #print(type(tf.split(pdparam, high - low + 1, axis=len(pdparam.get_shape()) - 1)))
        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = u.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = u.function([stochastic, ob], [ac, self.vpred])
        self._aux_act = u.function([stochastic, ob], [ac, self.pi_vpred, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, np.expand_dims(ob, 0))
        return ac1[0], vpred1[0]

    # This may not even be necessary, since the act function is only used in the trajectory generation, where the shared value is not needed
    def aux_act(self, stochastic, ob):
        ac1, ac_vpred, vpred1 = self._aux_act(stochastic, np.expand_dims(ob, 0))
        return ac1[0], ac_vpred[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []


def train(num_timesteps, seed, model_file, save_model_with_prefix, restore_model_from_file, save_after,
          load_after_iters, viz, stochastic, recording, test):
    rank = MPI.COMM_WORLD.Get_rank()
    sess = u.single_threaded_session()

    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    g = tf.get_default_graph()
    with g.as_default():
        tf.set_random_seed(workerseed)

    env = ProstheticsEnvMulticlip(visualize=viz, model_file=model_file, integrator_accuracy=1e-2)
    env_string = model_file[10:-5]
    destination_string = "../Results/"
    save_string = destination_string + env_string

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=312, num_hid_layers=2)

    env.seed(workerseed)
    pposgd_simple.learn(env,
                        workerseed,
                        policy_fn,
                        max_timesteps=int(num_timesteps * 1.1),
                        timesteps_per_actorbatch=1536,  # 1536
                        clip_param=0.2,
                        entcoeff=0.01,
                        optim_epochs=4,
                        optim_stepsize=1e-3,
                        optim_batchsize=512,
                        optim_aux=1e-3,
                        aux_batch_iters=4,
                        aux_optim_epochs=12,
                        gamma=0.999,
                        lam=0.9,
                        beta=0.00012,
                        aux_iters=64,
                        schedule='linear',
                        save_model_with_prefix=save_model_with_prefix,
                        dir_prefix=save_string,
                        save_prefix=env_string,
                        restore_model_from_file=restore_model_from_file,
                        load_after_iters=load_after_iters,
                        save_after=save_after,
                        stochastic=stochastic,
                        recording=recording,
                        test=test)
    env.close()


# args = ["mpirun", "-np", "4", "python", "main.py", "0", "model_file_name.osim", "training_data.csv"]
restore = int(sys.argv[1])
model_file = sys.argv[2]
test = sys.argv[3]
iteration = -1
if restore == 1:
    with open("../Results/" + model_file[10:-5] + '/iterations.txt', 'r') as f:
        lines = f.read().splitlines()
        iteration = int(lines[-1])

train(num_timesteps=5000000000,
      seed=990,
      model_file=model_file,
      save_model_with_prefix=True,
      restore_model_from_file=restore,
      save_after=25,
      load_after_iters=iteration,
      viz=True,
      stochastic=True,
      recording=True,
      test=test)