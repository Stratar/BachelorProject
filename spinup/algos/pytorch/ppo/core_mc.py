import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


'''class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a'''

class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.

        # The obs dimensions going in, is 300xstate_dim. 
        pi, val = self._distribution(obs)
        logp_a = None
        if act is not None:
            #print("In FORWARD\nOBS:\n", obs, "\n", len(obs))
            #print("pi:\n", pi)
            logp_a = self._log_prob_from_distribution(pi, act)
            logp_a = logp_a
            #print("logp_a:\n", logp_a)
        return pi, logp_a, val


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


'''class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

'''
class MultiCategoricalPdType:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.ncats = high - low + 1
    def pdclass(self):
        return MultiCategoricalPd
    def pdfromflat(self, flat):
        if(len(list((flat.size())))>1):
            pff = []
            for f in flat:
                #print("The current f in flat", f)
                #print("\nThe MultiCategoricalPd is:\n", MultiCategoricalPd(self.low, self.high, f))
                pff.append(MultiCategoricalPd(self.low, self.high, f))
                #MultiCategoricalPd(self.low, self.high, flat)
            return pff
        return MultiCategoricalPd(self.low, self.high, flat)
    def param_shape(self):
        return [sum(self.ncats)]
    def sample_shape(self):
        return [len(self.ncats)]
    def sample_dtype(self):
        return torch.int32

class CategoricalPd:
    def __init__(self, logits):
        #print("CategoricalPd logits")
        #print(logits)
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return torch.argmax(self.logits, dim=-1)
    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        one_hot_actions = torch.nn.functional.one_hot(x.to(torch.int64), list(self.logits.shape)[-1])
        #one_hot_actions = tf.one_hot(x, list(self.logits.get_shape())[-1])
        '''return tf.nn.softmax_cross_entropy_with_logits(
                                    logits=self.logits,
                                    labels=one_hot_actions)'''
        # Manual cross entropy because otherwise the dim is wrong and crashes
        #print(torch.rand(3,5))
        #print(self.logits)
        #print(one_hot_actions)
        lp = torch.nn.functional.log_softmax(self.logits, dim=-1)
        #print(lp)
        #print(torch.tensor([[lp[0]], [lp[1]]]))
        #loss = []
        #torch.tensor([[one_hot_actions[0], one_hot_actions[1], one_hot_actions[0], one_hot_actions[1]]])
        #for i in range(len(lp)):

        # There are errors with the dims of nll loss for the lp tensor, but when the one hot matches, that's not supported. 
        # Maybe search for multi-dimensional nll_loss
        loss = torch.nn.functional.nll_loss(torch.tensor([[lp[0], lp[1]], [lp[0], lp[1]]]), one_hot_actions)
        #loss = []


        ''' print(one_hot_actions[0])
                             print(torch.split(one_hot_actions, 1, dim=-1))
                             print(self.logits)
                             print(torch.split(self.logits, 1, dim=-1))
                             loss = torch.nn.functional.cross_entropy(torch.tensor([one_hot_actions[0], one_hot_actions[1]]), torch.tensor([self.logits[0], self.logits[1]]))'''
        #loss = torch.nn.CrossEntropyLoss(self.logits, one_hot_actions)
        return loss
    def kl(self, other):
        a0 = self.logits - torch.max(self.logits, dim=-1, keepdim=True)
        a1 = other.logits - torch.max(other.logits, dim=-1, keepdim=True)
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = torch.sum(ea0, dim=-1, keepdim=True)
        z1 = torch.sum(ea1, dim=-1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), dim=-1)
    def entropy(self):
        a0 = self.logits - torch.max(self.logits, dim=-1, keepdim=True)
        ea0 = torch.exp(a0)
        z0 = torch.sum(ea0, dim=-1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (torch.log(z0) - a0), dim=-1)
    def sample(self):
        u = torch.rand(self.logits.shape)
        return torch.argmax(self.logits - torch.log(-torch.log(u)), dim=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

'''class MultiCategoricalPd:
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = torch.tensor(low, dtype=torch.int32)
        #dim=len(flat.shape()) - 1
        print(len(high - low + 1))
        #split_param = []
        #for _ in range(len(high - low + 1)):
        #split_param.append(torch.split(flat, high - low + 1, dim=1))
        split_param = torch.split(flat, 2, dim=-1)
        self.categoricals = list(map(CategoricalPd, split_param))
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.low + torch.stack([p.mode() for p in self.categoricals], dim=-1).to(torch.int32)
    def neglogp(self, x):
        return torch.sum([p.neglogp(px) for p, px in zip(self.categoricals, torch.unbind(x - self.low, dim=len(x.shape) - 1))])
    
    def logp(self, x):
        return - self.neglogp(x)
    def kl(self, other):
        return torch.sum([
                p.kl(q) for p, q in zip(self.categoricals, other.categoricals)
            ])
    def entropy(self):
        return torch.sum([p.entropy() for p in self.categoricals])
    def sample(self):
        return self.low + torch.stack([p.sample() for p in self.categoricals], dim=-1).to(torch.int32)
    @classmethod
    def fromflat(cls, flat):
        raise NotImplementedError'''
class MultiCategoricalPd:
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = torch.tensor(low, dtype=torch.int32)
        #dim=len(flat.shape()) - 1
        #print(len(high - low + 1))
        #split_param = []
        #for _ in range(len(high - low + 1)):
        #split_param.append(torch.split(flat, high - low + 1, dim=1))
        # There could be an issue with how the param is split and how the categoricals are mapped!
        split_param = torch.split(flat, 2, dim=-1)
        self.categoricals = list(map(Categorical, split_param))
        #print("The current categoricals are:\n", self.categoricals)
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.low + torch.stack([p.mode() for p in self.categoricals], dim=-1).to(torch.int32)
    def neglogp(self, x):
        return torch.sum([p.neglogp(px) for p, px in zip(self.categoricals, torch.unbind(x - self.low, dim=len(x.shape) - 1))])
    
    def logp(self, x):
        #print("Inside the logp function: ")
        #print(x)
        lp_sum = 0
        for p in self.categoricals :
            #print("Current Categorical is: ")
            #print(p)
            lp = p.log_prob(x)
            #print("lp:\n", lp.mean())
            # Using this the logps look closer to the logps from the gaussian dist
            lp_sum += lp.mean()
            #np.sum(p.log_prob(x) for p in self.categoricals)
        #print("OUT OF THE LOOP")
        return lp_sum
    def kl(self, other):
        return torch.sum([
                p.kl(q) for p, q in zip(self.categoricals, other.categoricals)
            ])
    def entropy(self):
        return np.sum([p.entropy() for p in self.categoricals])
    def sample(self):
        return self.low + torch.stack([p.sample() for p in self.categoricals], dim=-1).to(torch.int32)
    @classmethod
    def fromflat(cls, flat):
        raise NotImplementedError

class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, low, high):
        super().__init__()
        #log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        #self.log_std = torch.nn.Parameter(torch.as_tensor(log_std)).to(device)
        self.low = low
        self.high = high
        self.pdtype = pdtype = MultiCategoricalPdType(low, high)

        self.nn_layer = nn.Sequential(
                nn.Linear(obs_dim, hidden_sizes[0]),
                nn.Tanh(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.Tanh()
              ).float()

        self.actor_layer = nn.Sequential(
                nn.Linear(hidden_sizes[1], 36),
                nn.Sigmoid()
              ).float()

        self.critic_layer = nn.Sequential(
                nn.Linear(hidden_sizes[1], 1)
              ).float()
        self.to(device)

    def _distribution(self, obs):
        x = self.nn_layer(obs)
        mu = self.actor_layer(x)
        val = torch.squeeze(self.critic_layer(x), -1)
        #print("In the _distribution, mu:\n", mu, "with size:\n", len(list(mu.size())))
        cat_pd = self.pdtype.pdfromflat(mu)
        #ac = cat_pd.sample()
        #print(ac)
        #std = torch.exp(self.log_std).to(device)
        #Normal(mu, std)
        return cat_pd, val

    def _log_prob_from_distribution(self, pi, act):
        #print(act)
        if type(pi) is list:
            logs = []
            for idx in range(len(pi)):
                logs.append(pi[idx].logp(act[idx]).sum(axis=-1).tolist())
            return torch.tensor(logs, dtype=torch.float, requires_grad=True)
        #print("Pi lenght in _log_prob_from_distribution:\n",pi, len(pi))
        return pi.logp(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    '''    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    '''
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = nn.Sequential(
                nn.Linear(obs_dim, hidden_sizes[0]),
                nn.Tanh(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.Tanh(),
                nn.Linear(hidden_sizes[1], 1)
              ).float()
        self.to(device)

    def forward(self, obs):
        state = torch.tensor(obs, dtype=torch.float).to(device)
        return torch.squeeze(self.v_net(state), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, low, high,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation, low, high)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        state = torch.tensor(obs, dtype=torch.float).to(device)
        with torch.no_grad():
            pi, _ = self.pi._distribution(state)
            a = pi.sample()
            '''print("********Data in Step: ")
                                                print("Input observation is: ")
                                                print(state)
                                                print("sampled action: ")
                                                print(a)'''
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            '''print("logp_a")
                                                print(logp_a)
                                                print("**********************")'''
            v = self.v(state)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]
