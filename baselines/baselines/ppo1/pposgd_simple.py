import os
import time
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from mpi4py import MPI
import baselines.common.tf_util as u
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
from baselines.common.mpi_adam import MpiAdam
from collections import deque
import pickle


def traj_segment_generator(networks, env, horizon, stochastic, recording=False):#networks is pi
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset(test=False, record=recording)
    #ob = env.reset(test=False)

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    cur_ep_true_ret = 0
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...
    ep_true_rets = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    true_rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        # Change this to call a function if we want to test a certain behavior
        ac, vpred = networks.act(stochastic, ob)
        

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets,
                   "true_rews": true_rews}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_true_rets = []

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        ob, rew, true_rew, new = env.step(ac)

        rews[i] = rew
        true_rews[i] = true_rew

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew

        cur_ep_len += 1
        if new:
            '''
            logger.log(cur_ep_ret)
            logger.log(cur_ep_true_ret)
            logger.log(cur_ep_len)
            logger.log("----------------New Episode----------------")
            base_path = os.path.dirname(os.path.abspath(__file__))
            f = open(base_path + "/../../../logs.txt", "a+")
            f.write("----------------New Episode----------------\n")
            f.close()
            '''
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            ep_true_rets.append(cur_ep_true_ret)
            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_true_ret = 0

            ob = env.reset(test=False, record=recording)
            #ob = env.reset(test=False)

        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, seed, policy_fn, *,
          timesteps_per_actorbatch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize, optim_aux, aux_batch_iters, aux_optim_epochs, # optimization hypers
          gamma, lam,  # advantage estimation
          beta, aux_iters, # after how many ppo updates it'll do the auxiliary phase
          save_model_with_prefix,  # Save the model
          dir_prefix,
          save_prefix,
          restore_model_from_file,  # Load the states/model from this file.
          load_after_iters,
          save_after,
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
          stochastic=True,
          recording=False,
          test=0
          ):
    ob_space = env.observation_space
    ac_space = env.action_space

    g = tf.get_default_graph()
    with g.as_default():
        tf.set_random_seed(seed)

    networks = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy
    oldnetworks = policy_fn("oldpi", ob_space, ac_space)  # Network for old policy

    #pi = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy
    #oldpi = policy_fn("oldpi", ob_space, ac_space)  # Network for old policy

    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return
    true_ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return from the shared network

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32,
                            shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    ob = u.get_placeholder_cached(name="ob")
    ac = networks.pi.pdtype.sample_placeholder([None])
    #ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldnetworks.pi.pd.kl(networks.pi.pd)
    #kloldnew = oldpi.pd.kl(pi.pd)
    ent = networks.pi.pd.entropy()
    #ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(networks.pi.pd.logp(ac) - oldnetworks.pi.pd.logp(ac))  # pnew / pold
    #ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)

    vf_loss = tf.reduce_mean(tf.square(networks.val.vpred - ret))
    #vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen
    #total_loss = pol_surr + pol_entpen + vf_loss
    pol_losses = [pol_surr, pol_entpen, meankl, meanent]
    val_losses = [vf_loss]
    #losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]
    log_loss_names = ["pol_surr", "pol_entpen", "kl", "ent"]

    # Get trainable variables is a custom function in main.py in the MLP class, EDIT to get specific variable scopes
    #...if you want to make separate adam optimisers
    pol_var_list = networks.pi.get_trainable_variables()
    val_var_list = networks.val.get_trainable_variables()
    #var_list = pi.get_trainable_variables()

    # The auxiliary buffer being created
    aux_dict = {"ob": [], "ac": [], "vtarg": [], "true": []}
    dict_size = 0 # Used to determine the size of the aux buffer, for the batch size

    #logger.log(pi.get_trainable_variables(scope="pi/vf"))
    if aux_iters != 0:
        # Adding the Aux specific calculations
        
        aux_meankl = tf.math.reduce_mean(oldnetworks.pi.pd.kl(networks.pi.pd))
        #aux_meankl = tf.math.reduce_mean(oldpi.pd.kl(pi.pd))
        aux_loss = tf.reduce_mean((1 - tf.math.exp(-beta/true_ret)) * tf.reduce_mean(tf.square(networks.pi.vpred - ret))) #tf.reduce_mean() over the whole equation
        #aux_loss = tf.reduce_mean((1 - tf.math.exp(-beta/true_ret)) * tf.reduce_mean(tf.square(pi.pi_vpred - ret))) #tf.reduce_mean() over the whole equation
        joint_loss = aux_loss + aux_meankl
        # Adding in the same backward loss, the vf loss
        aux_total_loss = joint_loss + vf_loss
        auxlosses = [aux_meankl, aux_loss, vf_loss]

        # Imitating the way ppo lossandgrad is built, with the inputs being the components of the calculations
        #...auxlosses the calculation variables and the flatgrad the total loss that is made up of the calculations
        #...as well as the variable list of the networks. 
        # Since PPG paper asks for the extra value update after the aux update, maybe I sould add it to this, or
        #...make a second one for the value loss
        auxlossandgrad = u.function([ob, ac, ret, true_ret], auxlosses + [u.flatgrad(aux_total_loss, pol_var_list)])
        #auxlossandgrad = u.function([ob, ac, ret, true_ret], auxlosses + [u.flatgrad(aux_total_loss, var_list)])

    pol_lossandgrad = u.function([ob, ac, atarg, lrmult], pol_losses + [u.flatgrad(total_loss, pol_var_list)])
    val_lossandgrad = u.function([ob, ret], val_losses + [u.flatgrad(vf_loss, val_var_list)])
    #lossandgrad = u.function([ob, ac, atarg, ret, lrmult], losses + [u.flatgrad(total_loss, var_list)])
    pol_adam = MpiAdam(pol_var_list, epsilon=adam_epsilon)
    val_adam = MpiAdam(val_var_list, epsilon=adam_epsilon)
    #adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = u.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldnetworks.pi.get_variables(), networks.pi.get_variables())])    
    '''
    assign_old_eq_new = u.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])
    '''
    compute_losses = u.function([ob, ac, atarg, lrmult], pol_losses)
    #compute_losses = u.function([ob, ac, atarg, ret, lrmult], losses)

    u.initialize()
    pol_adam.sync()
    val_adam.sync()
    #adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(networks, env, timesteps_per_actorbatch, stochastic=stochastic, recording=recording)
    #seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=stochastic, recording=recording)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    iters_this_run = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards
    truerewbuffer = deque(maxlen=100)

    if restore_model_from_file == 1:
        saver = tf.train.Saver()
        base_path = os.path.dirname(os.path.abspath(__file__))
        logger.log(save_prefix)
        model_f = os.path.normpath(base_path +
                                   "/../../../" +
                                   dir_prefix +
                                   "/models/" +
                                   save_prefix +
                                   "_afterIter_" +
                                   str(load_after_iters) +
                                   ".model")
        logger.log(model_f)
        saver.restore(tf.get_default_session(), model_f)
        logger.log("Loaded model from {}".format(model_f))
        # Restore the variables from file
        data = genfromtxt(dir_prefix + '/test_afterIter_' + str(load_after_iters) + '.csv', delimiter=',')
        for i in range(len(data)):
            data_vector = data[i]
            episodes_so_far = int(data_vector[0])
            timesteps_so_far = int(data_vector[1])
            iters_so_far = int(data_vector[2])
            time_elapsed = int(data_vector[3])
            lenbuffer.append(int(data_vector[4]))
            rewbuffer.append(int(data_vector[5]))
            truerewbuffer.append(int(data_vector[6]))
        if aux_iters != 0:
            buf_file = open(dir_prefix + '/buffers/aux_buffer_iter_' + str(iters_so_far) + '.npy', "rb")
            aux_dict = pickle.load(buf_file)
            buf_file.close()
            dict_size = (len(aux_dict["ob"]) + len(aux_dict["ac"]) + len(aux_dict["vtarg"]) +len(aux_dict["true"]))/4

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    max_time = 0 # Value holding the longest iteration for safety restart save
    while True:
        iter_tstart = time.time()
        if callback:
            callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)
        
        '''logger.log(var_list)
                                for item in var_list:
                                    logger.log(item.value)'''
        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        ob, ac, atarg, tdlamret, true = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["true_rews"]

        vpredbefore = seg["vpred"]  # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not networks.pi.recurrent)
        #d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(networks.pi, "ob_rms"):
            networks.pi.ob_rms.update(ob)  # update running mean/std for policy
        #if hasattr(pi, "ob_rms"):
        #    pi.ob_rms.update(ob)  # update running mean/std for policy

        assign_old_eq_new()  # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))

        for _ in range(optim_epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = pol_lossandgrad(batch["ob"], batch["ac"], batch["atarg"], cur_lrmult)
                #*newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                pol_adam.update(g, optim_stepsize * cur_lrmult)
                #adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")

        for _ in range(optim_epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = val_lossandgrad(batch["ob"], batch["vtarg"])
                #*newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                val_adam.update(g, optim_stepsize * cur_lrmult)
                #adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []

        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], cur_lrmult)
            #newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses = np.mean(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, log_loss_names):
        #for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_" + name, lossval)

        # Add the relevant data to the auxiliary buffer
        if aux_iters != 0:
            # This is meant to control the size of the buffer to a manageable size, as well as benefit from the most recent experiences of the model

            if dict_size >= timesteps_per_actorbatch*aux_iters/4.0:
                del aux_dict["ob"][:len(ob.tolist())]
                del aux_dict["ac"][:len(ac.tolist())]
                del aux_dict["vtarg"][:len(tdlamret.tolist())]
                del aux_dict["true"][:len(true.tolist())]
            
            aux_dict["ob"] += ob.tolist()
            aux_dict["ac"] += ac.tolist()
            aux_dict["vtarg"] += tdlamret.tolist()
            aux_dict["true"] += true.tolist()
            logger.log(np.mean(aux_dict["true"]))
            dict_size = (len(aux_dict["ob"]) + len(aux_dict["ac"]) + len(aux_dict["vtarg"]) +len(aux_dict["true"]))/4
            logger.log(dict_size)

        # Adding the auxiliary phase after all the updates for the ppo have been done
        if aux_iters != 0 and (iters_so_far % aux_iters == 0) and (iters_so_far is not 0):
            logger.log("*Auxiliary Phase*")
            d_aux = Dataset(dict(ob=np.array(aux_dict["ob"]), ac=np.array(aux_dict["ac"]), vtarg=np.array(aux_dict["vtarg"]), true=np.array(aux_dict["true"])), 
                                                                                                                                    shuffle=not networks.pi.recurrent)
            '''d_aux = Dataset(dict(ob=np.array(aux_dict["ob"]), ac=np.array(aux_dict["ac"]), vtarg=np.array(aux_dict["vtarg"]), true=np.array(aux_dict["true"])), 
                                                                                                                                                                        shuffle=not pi.recurrent)'''
            for _ in range(aux_optim_epochs):
                aux_losses = []  # list of tuples, each of which gives the loss for a minibatch
                for batch in d_aux.iterate_once(int(dict_size/aux_batch_iters)):
                    *newlosses, g = auxlossandgrad(batch["ob"], batch["ac"], batch["vtarg"], batch["true"])
                    pol_adam.update(g, optim_aux * cur_lrmult)
                    #adam.update(g, optim_aux * cur_lrmult)
                    aux_losses.append(newlosses)
                logger.log(fmt_row(13, np.mean(aux_losses, axis=0)))

            aux_dict.clear()
            aux_dict = {"ob": [], "ac": [], "vtarg": [], "true": []}
            dict_size = 0


        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews, truerews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        truerewbuffer.extend(truerews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(truerewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        prev_episodes_so_far = episodes_so_far
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)

        iters_this_run += 1
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - iter_tstart)
        logger.record_tabular("TimeElapsedMean", (time.time() - tstart) / iters_this_run)
        logger.record_tabular("TimeElapsedTotal", time.time() - tstart)

        # Store the longest that an iteration has taken so far, in order to save before restart
        if time.time() - iter_tstart > max_time:
            max_time = time.time() - iter_tstart

        if MPI.COMM_WORLD.Get_rank() == 0 and test == 0:
            f = open(dir_prefix + "/training_rewards.txt", "a+")
            g = open(dir_prefix + "/training_episode_lengths.txt", "a+")
            h = open(dir_prefix + "/training_mean_rewards.txt", "a+")
            k = open(dir_prefix + "/training_mean_lengths.txt", "a+")
            l = open(dir_prefix + "/iterations.txt", "a+")
            m = open(dir_prefix + "/timesteps.txt", "a+")
            n = open(dir_prefix + "/training_mean_truerewards.txt", "a+")
            h.write("Episode %d    " % episodes_so_far)
            try:
                h.write("Reward  %d\r\n" % np.mean(rews))
            except ValueError as e:
                h.write("Reward  %d\r\n" % 0)
            k.write("Episode %d    " % episodes_so_far)
            k.write("Length  %d\r\n" % np.mean(lens))
            n.write("Episode %d    " % episodes_so_far)
            n.write("Reward  %d\r\n" % np.mean(truerews))
            if iters_so_far % save_after == 0 or 10800 - (time.time() - tstart) <= max_time:
                l.write("%d\r\n" % iters_so_far)
            m.write("%d\r\n" % timesteps_so_far)
            for i in range(episodes_so_far - prev_episodes_so_far):
                f.write("Episode %d    " % (prev_episodes_so_far + i))
                try:
                    f.write("Reward  %d\r\n" % rews[i])
                except ValueError as e:
                    f.write("Reward  %d\r\n" % 0)
                g.write("Episode %d    " % (prev_episodes_so_far + i))
                g.write("Length  %d\r\n" % lens[i])
            f.close()
            g.close()
            k.close()
            h.close()
            l.close()
            m.close()
            n.close()

            logger.dump_tabular()

        if (iters_so_far % save_after == 0 or 10800 - (time.time() - tstart) <= max_time) and test == 0:
            if save_model_with_prefix:
                base_path = os.path.dirname(os.path.abspath(__file__))
                model_f = os.path.normpath(base_path +
                                           "/../../../" +
                                           dir_prefix +
                                           "/models/" +
                                           save_prefix +
                                           "_afterIter_" +
                                           str(iters_so_far) +
                                           ".model")
                u.save_state(model_f)
                logger.log("Saved model to file :{}".format(model_f))
                if episodes_so_far < 100:
                    size = episodes_so_far
                else:
                    size = 100
                asd = np.zeros((size, 7), dtype=np.int32)
                for i in range(size):
                    asd[i] = [episodes_so_far, timesteps_so_far, iters_so_far, time.time() - tstart, lenbuffer[i],
                              rewbuffer[i], truerewbuffer[i]]
                    np.savetxt(dir_prefix + '/test_afterIter_' + str(iters_so_far) + '.csv', asd, delimiter=",")

                # Storing the aux Buffer
                #Maybe store as np.save and load with np.load?
                buf_file = open(dir_prefix + '/buffers/aux_buffer_iter_' + str(iters_so_far) + '.npy', "wb")
                pickle.dump(aux_dict, buf_file)
                buf_file.close()

    return pi


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]