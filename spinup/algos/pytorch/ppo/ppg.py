import numpy as np
from numpy import genfromtxt
import torch
import os
from osim.env import ProstheticsEnvMulticlip
from copy import deepcopy
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



def ppg(model_file, load_after_iters, restore_model_from_file=1, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=40, train_v_iters=40, train_aux_iters=40, aux_iters=2, 
        lam=0.97, max_ep_len=1000, target_kl=0.01, logger_kwargs=dict(), save_freq=2, viz=False):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    save_prefix = model_file[10:-5]
    # Instantiate environment
    #env = env_fn()

    # Use this for OpenSim  
    env = ProstheticsEnvMulticlip(visualize=viz, model_file=model_file, integrator_accuracy=1e-2)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # Use this for gym
    #model_file = 'Pendulum-v0'
    #env = gym.make(model_file)
    #obs_dim = env.observation_space.shape
    #act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Initialise the data in order to be able to load it
    episodes = 0
    n_steps = 0
    start_epoch = 0


    def save_weights(save_epoch):
        policy_path = save_epoch + "policy.tar"
        torch.save({
            'model_state_dict': ac.pi.state_dict(),
            'optimizer_state_dict': pi_optimizer.state_dict()
            }, policy_path)
        
        value_path = save_epoch + "value.tar"
        torch.save({
            'model_state_dict': ac.v.state_dict(),
            'optimizer_state_dict': vf_optimizer.state_dict()
            }, value_path)
        
    def load_weights(load_epoch):
        policy_path = load_epoch + "policy.tar"
        policy_checkpoint = torch.load(policy_path)
        ac.pi.load_state_dict(policy_checkpoint['model_state_dict'])
        pi_optimizer.load_state_dict(policy_checkpoint['optimizer_state_dict'])

        value_path = load_epoch + "value.tar"
        value_checkpoint = torch.load(value_path)
        ac.v.load_state_dict(value_checkpoint['model_state_dict'])
        vf_optimizer.load_state_dict(value_checkpoint['optimizer_state_dict'])


    if restore_model_from_file == 1:
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_f = os.path.normpath(
            base_path + "/../../../../" + save_prefix + '/models/' + save_prefix + "_afterIter_" + str(
                load_after_iters))    
        load_weights(model_f)
        # Restore the variables from file
        data = genfromtxt(save_prefix + '/test_afterIter_' + str(load_after_iters) + '.csv', delimiter=',')
        num_rows = sum(1 for row in data)
        for i in range(len(data)):
            data_vector = data[i]
            episodes = int(data_vector[0])
            n_steps = int(data_vector[1])
            start_epoch = int(data_vector[2])
    # Sync params across processes

    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        obs = torch.tensor(obs, dtype=torch.float).to(device)
        act = torch.tensor(act, dtype=torch.float).to(device)
        adv = torch.tensor(adv, dtype=torch.float).to(device)
        logp_old = torch.tensor(logp_old, dtype=torch.float).to(device)
        # Policy loss
        pi, logp, _ = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        obs = torch.tensor(obs, dtype=torch.float).to(device)
        ret = torch.tensor(ret, dtype=torch.float).to(device)
        return ((ac.v(obs) - ret)**2).mean()

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    #aux_data = dict() # Store the ppo buffer states
    def update():
        data = buf.get()
        #Maybe do this only when there is an upcoming aux update
        aux_data = deepcopy(data)
        #print(aux_data['obs'])
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
        return aux_data

    def compute_aux_loss(aux_data):
        #print(aux_data)
        obs, act, logp_old = aux_data['obs'], aux_data['act'], aux_data['logp']

        obs = torch.tensor(obs, dtype=torch.float).to(device)
        act = torch.tensor(act, dtype=torch.float).to(device)
        logp_old = torch.tensor(logp_old, dtype=torch.float).to(device)

        ret = ac.v(obs)
        pi, logp, val = ac.pi(obs, act)
        aux_loss = ((ret - val)**2).mean()
        loss = torch.nn.KLDivLoss(size_average=False)(logp, logp_old) + aux_loss
        return loss

    def aux_update(aux_data):
        for _ in range(train_aux_iters):
            pi_optimizer.zero_grad()
            loss = compute_aux_loss(aux_data)
            loss.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()


    # Prepare for interaction with environment
    start_time = time.time()
    # Use this for gym
    #o, ep_ret, ep_len, true_reward = env.reset(), 0, 0, 0
    # Use this for OpenSim
    o, ep_ret, ep_len = env.reset(test=False), 0, 0

    base_path = os.path.dirname(os.path.abspath(__file__))
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(start_epoch, epochs):
        ret_arr = []
        true_arr = []
        logger.log("********** Iteration %i ************" % epoch)
        ep_true = 0
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            # Use this for OpenSim
            next_o, r, true_reward, d = env.step(a)

            # Use this for gym
            #next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            ep_true += true_reward

            # save and log
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    ret_arr.append(ep_ret)
                    true_arr.append(ep_true)
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpTRew=ep_true)
                    ep_true = 0

                    episodes += 1
                n_steps += ep_len
                # Use this for OpenSim
                o, ep_ret, ep_len, true_reward = env.reset(test=False), 0, 0, 0
                # Use this for gym
                #o, ep_ret, ep_len, true_reward = env.reset(), 0, 0, 0

        logger.store(EpTrue=np.mean(true_arr), Episodes=episodes)

        # Save model
        '''if (epoch % save_freq == 0) or (epoch == epochs-1):
                                    logger.save_state({'env': env}, None)
                        '''
        # Perform PPO update!
        aux_data = update()
        if epoch%aux_iters==0 and epoch!=0 :
            aux_update(aux_data)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        #logger.log_tabular('VVals', with_min_and_max=True)
        #logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        '''logger.log_tabular('Entropy', average_only=True)
                                logger.log_tabular('KL', average_only=True)
                                logger.log_tabular('ClipFrac', average_only=True)
                                logger.log_tabular('StopIter', average_only=True)'''
        logger.log_tabular('Episodes')
        logger.log_tabular('EpTrue', np.mean(true_arr))
        logger.log_tabular('EpTRew', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

        #if proc_id == 0:
        if proc_id() == 0:
            l = open(save_prefix + "/iterations.txt", "a+")
            m = open(save_prefix + "/timesteps.txt", "a+")                                                                  
            n = open(save_prefix + "/training_mean_truerewards.txt", "a+")                                                                  
            r = open(save_prefix + "/training_mean_rewards.txt", "a+")
            
            n.write("Episode %d    " % episodes)
            n.write("Reward  %d\r\n" % np.mean(true_arr))

            r.write("Episode %d    " % episodes)
            r.write("Reward  %d\r\n" % np.mean(ret_arr))

            #ALSO STORE THE TIMESTEPS SO IT CAN STOP AND RESTART PROPERLY
            if epoch % save_freq == 0:
                l.write("%d\r\n" % epoch)

            m.write("%d\r\n" % n_steps)
            
            l.close()
            m.close()
            n.close()
            r.close()
            #It has been indented once
            if epoch % save_freq == 0:
            #if save_model_with_prefix:
                base_path = os.path.dirname(os.path.abspath(__file__))
                model_f = os.path.normpath(
                    base_path + '/../../../../' + save_prefix + '/models/' + save_prefix + "_afterIter_" + str(
                        epoch))
                #Use the model_f as destination location, so change the save_models()
                #save_actor_model(model_f)
                #save_critic_model(model_f)
                save_weights(model_f)
                logger.log("Saved model to file :{}".format(model_f))
                if episodes < 100:
                    size = episodes
                else:
                    size = 100
                asd = np.zeros((size, 4), dtype=np.int32)
                for i in range(size):
                    asd[i] = [episodes, n_steps, epoch, np.mean(true_arr)]
                    np.savetxt(save_prefix + '/test_afterIter_' + str(epoch) + '.csv', asd, delimiter=",")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('restore', type=int, default=1)
    parser.add_argument('load_iters', type=int, default=1)
    parser.add_argument('env', type=str, default='BipedalWalker-v3')
    parser.add_argument('cpu', type=int, default=1)
    parser.add_argument('--hid', type=int, default=312)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1536)#4000#1536
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    load_iters = args.load_iters
    if args.load_iters == 1:
            with open(args.env[10:-5] + '/iterations.txt', 'r') as f:
                lines = f.read().splitlines()
                last_iter = int(lines[-1])
                load_iters = last_iter

    ppg(model_file=args.env, load_after_iters=load_iters, 
        restore_model_from_file=args.restore, actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)