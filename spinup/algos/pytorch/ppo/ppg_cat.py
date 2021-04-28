import os
from collections import deque, namedtuple

import os
from tqdm import tqdm
import numpy as np
from numpy import genfromtxt
import torch
from osim.env import ProstheticsEnvMulticlip
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

import gym

# constants

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data

Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward', 'done', 'value'])
AuxMemory = namedtuple('Memory', ['state', 'target_value', 'old_values'])

class MultiCategoricalPdType(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.ncats = high - low + 1
    def pdclass(self):
        return MultiCategoricalPd
    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.low, self.high, flat)
    def param_shape(self):
        return [sum(self.ncats)]
    def sample_shape(self):
        return [len(self.ncats)]
    def sample_dtype(self):
        return torch.int32

class CategoricalPd(object):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return torch.argmax(self.logits, dim=-1)
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
        u = torch.rand(self.logits.shape())
        return torch.argmax(self.logits - torch.log(-torch.log(u)), dim=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class MultiCategoricalPd(object):
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = torch.tensor(low, dtype=tf.int32)

        split_param = torch.split(pdparam, high - low + 1, axis=len(pdparam.get_shape()) - 1)

        self.categoricals = list(map(CategoricalPd, split_param))
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.low + torch.stack([p.mode() for p in self.categoricals], dim=-1).to(torch.int32)
    def neglogp(self, x):
        return torch.sum([p.neglogp(px) for p, px in zip(self.categoricals, torch.unbind(x - self.low, dim=len(x.get_shape()) - 1))])
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
        raise NotImplementedError

class ExperienceDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))

def create_shuffled_dataloader(data, batch_size):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size = batch_size, shuffle = True)

# helpers

def exists(val):
    return val is not None

def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

def init_(m):
    if isinstance(m, nn.Linear):
        gain = torch.nn.init.calculate_gain('tanh')
        torch.nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# networks

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions, low, high):
        super().__init__()
        self.low = low
        self.high = high
        self.pdtype = pdtype = MultiCategoricalPdType(low, high)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Linear(hidden_dim, 1)
        self.apply(init_)

    def forward(self, x):
        hidden = self.net(x)
        action_layer = self.action_head(hidden)
        pdparam = nn.Linear(action_layer, self.pdtype.param_shape()[0])
        self.pd = pdtype.pdfromflat(pdparam)
        ac = self.pd.sample()
        return ac, self.value_head(hidden)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_)

    def forward(self, x):
        return self.net(x)

# agent

def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))

class PPG:
    def __init__(
        self,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        value_clip,
        low,
        high.
    ):
        self.actor = Actor(state_dim, actor_hidden_dim, num_actions, low, high).to(device)
        self.critic = Critic(state_dim, critic_hidden_dim).to(device)
        self.opt_actor = Adam(self.actor.parameters(), lr=lr, betas=betas)
        self.opt_critic = Adam(self.critic.parameters(), lr=lr, betas=betas)

        self.minibatch_size = minibatch_size

        self.epochs = epochs
        self.epochs_aux = epochs_aux

        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s

        self.eps_clip = eps_clip
        self.value_clip = value_clip

    def save(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, f'./ppg.pt')

    def load(self):
        if not os.path.exists('./ppg.pt'):
            return

        data = torch.load(f'./ppg.pt')
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])

    def learn(self, memories, aux_memories, next_state):
        # retrieve and prepare data from memory for training
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        masks = []
        values = []

        for mem in memories:
            states.append(mem.state)
            actions.append(torch.tensor(mem.action))
            old_log_probs.append(mem.action_log_prob)
            rewards.append(mem.reward)
            masks.append(1 - float(mem.done))
            values.append(mem.value)

        # calculate generalized advantage estimate
        next_state = torch.from_numpy(next_state).to(device)
        next_value = self.critic(next_state).detach()
        values = values + [next_value]

        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lam * masks[i] * gae
            returns.insert(0, gae + values[i])

        # convert values to torch tensors
        to_torch_tensor = lambda t: torch.stack(t).to(device).detach()

        states = to_torch_tensor(states)
        actions = to_torch_tensor(actions)
        old_values = to_torch_tensor(values[:-1])
        old_log_probs = to_torch_tensor(old_log_probs)

        rewards = torch.tensor(returns).float().to(device)

        # store state and target values to auxiliary memory buffer for later training
        aux_memory = AuxMemory(states, rewards, old_values)
        aux_memories.append(aux_memory)

        # prepare dataloader for policy phase training
        dl = create_shuffled_dataloader([states, actions, old_log_probs, rewards, old_values], self.minibatch_size)

        # policy phase training, similar to original PPO
        for _ in range(self.epochs):
            for states, actions, old_log_probs, rewards, old_values in dl:
                action_probs, _ = self.actor(states)
                values = self.critic(states)
                dist = Categorical(action_probs)
                action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                # calculate clipped surrogate objective, classic PPO loss
                ratios = (action_log_probs - old_log_probs).exp()
                advantages = normalize(rewards - old_values.detach())
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - torch.min(surr1, surr2) - self.beta_s * entropy

                update_network_(policy_loss, self.opt_actor)

                # calculate value loss and update value network separate from policy network
                value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)

                update_network_(value_loss, self.opt_critic)

    def learn_aux(self, aux_memories):
        # gather states and target values into one tensor
        states = []
        rewards = []
        old_values = []
        for state, reward, old_value in aux_memories:
            states.append(state)
            rewards.append(reward)
            old_values.append(old_value)

        states = torch.cat(states)
        rewards = torch.cat(rewards)
        old_values = torch.cat(old_values)

        # get old action predictions for minimizing kl divergence and clipping respectively
        old_action_probs, _ = self.actor(states)
        old_action_probs.detach_()

        # prepared dataloader for auxiliary phase training
        dl = create_shuffled_dataloader([states, old_action_probs, rewards, old_values], self.minibatch_size)

        # the proposed auxiliary phase training
        # where the value is distilled into the policy network, while making sure the policy network does not change the action predictions (kl div loss)
        for epoch in range(self.epochs_aux):
            for states, old_action_probs, rewards, old_values in tqdm(dl, desc=f'auxiliary epoch {epoch}'):
                action_probs, policy_values = self.actor(states)
                action_logprobs = action_probs.log()

                # policy network loss copmoses of both the kl div loss as well as the auxiliary loss
                aux_loss = clipped_value_loss(policy_values, rewards, old_values, self.value_clip)
                loss_kl = F.kl_div(action_logprobs, old_action_probs, reduction='batchmean')
                policy_loss = aux_loss + loss_kl

                update_network_(policy_loss, self.opt_actor)

                # paper says it is important to train the value network extra during the auxiliary phase
                values = self.critic(states)
                value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)

                update_network_(value_loss, self.opt_critic)

# main

def main(
    env_name = 'LunarLander-v2',
    num_episodes = 50000,
    max_timesteps = 1536,
    actor_hidden_dim = 312,
    critic_hidden_dim = 312,
    minibatch_size = 512,
    lr = 0.0005,
    betas = (0.9, 0.999),
    lam = 0.95,
    gamma = 0.99,
    eps_clip = 0.2,
    value_clip = 1.75,
    beta_s = .01,
    update_timesteps = 5000,
    num_policy_updates_per_aux = 48,
    epochs = 4,
    epochs_aux = 6,
    seed = None,
    render = False,
    render_every_eps = 250,
    save_every = 2,
    load = False,
    monitor = False,
    viz = False
):
    #env = gym.make(env_name)
    env = ProstheticsEnvMulticlip(visualize=viz, model_file=env_name, integrator_accuracy=1e-2)
    #obs_dim = env.observation_space.shape[0]
    #act_dim = env.action_space.shape[0]

    if monitor:
        env = gym.wrappers.Monitor(env, './tmp/', force=True)

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    #num_actions = env.action_space.n
    print(env.action_space.shape[0])
    #print(num_actions)

    memories = deque([])
    aux_memories = deque([])

    agent = PPG(
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        value_clip,
        np.zeros_like(env.action_space.low),
        np.ones_like(env.action_space.high)
    )

    if load:
        agent.load()

    if exists(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    time = 0
    updated = False
    num_policy_updates = 0

    ########################################################################
    # Maybe add dtype=np.int32 in np like second argument
    param_shape = high - low + 1
    ########################################################################
    for eps in tqdm(range(num_episodes), desc='episodes'):
        render_eps = render and eps % render_every_eps == 0
        state = env.reset(test=False)
        ep_true = 0
        true_arr = []
        for timestep in range(max_timesteps):
            time += 1

            if updated and render_eps:
                env.render()

            state = torch.from_numpy(np.array(state)).to(device)
            action_probs, _ = agent.actor(torch.as_tensor(state, dtype=torch.float32))
            value = agent.critic(torch.as_tensor(state, dtype=torch.float32))
            ########################################################################
            pdparam = torch.Linear(action_probs, param_shape)

            split_param = torch.split(pdparam, high - low + 1, axis=len(pdparam.get_shape()) - 1)

            categoricals = list(map(CategoricalPd, split_param))
            ac = pd.sample()



            ########################################################################
            dist = Categorical(action_probs)
            action = dist.sample()
            #print(action_probs)
            action_log_prob = dist.log_prob(action)
            action = action.item()

            #next_state, reward, done, _ = env.step(action)
            #THIS IS PROBABLY WRONG!!!!!!!!
            next_state, reward, true_reward, done = env.step(action_probs.detach().numpy())
            ep_true += true_reward
            memory = Memory(state, action, action_log_prob, reward, done, value)
            memories.append(memory)

            state = next_state

            if time % update_timesteps == 0:
                agent.learn(memories, aux_memories, next_state)
                num_policy_updates += 1
                memories.clear()

                if num_policy_updates % num_policy_updates_per_aux == 0:
                    agent.learn_aux(aux_memories)
                    aux_memories.clear()

                updated = True

            if done:
                if render_eps:
                    updated = False
                break

        if render_eps:
            env.close()

        if eps % save_every == 0:
            agent.save()

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
    parser.add_argument('--steps', type=int, default=384)#4000#1536
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()


    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    load_iters = args.load_iters
    if args.load_iters == 1:
            with open(args.env[10:-5] + '/iterations.txt', 'r') as f:
                lines = f.read().splitlines()
                last_iter = int(lines[-1])
                load_iters = last_iter
    main(env_name=args.env)