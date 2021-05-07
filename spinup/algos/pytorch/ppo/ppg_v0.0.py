import gym
import os
from gym.envs.registration import register

from osim.env import ProstheticsEnvMulticlip
from numpy import genfromtxt
    
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy
import time
import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Utils():
    def prepro(self, I):
        I           = I[35:195] # crop
        I           = I[::2,::2, 0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0]   = 1 # everything else (paddles, ball) just set to 1
        X           = I.astype(np.float32).ravel() # Combine items in 1 array 
        return X

class Policy_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy_Model, self).__init__()

        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 312),
                nn.ReLU(),
                nn.Linear(312, 312),
                nn.ReLU()
              ).float().to(device)

        self.actor_layer = nn.Sequential(
                nn.Linear(312, action_dim),
                nn.Tanh()
              ).float().to(device)

        self.critic_layer = nn.Sequential(
                nn.Linear(312, 1)
              ).float().to(device)
        
    def forward(self, states):
        x = self.nn_layer(states)
        return self.actor_layer(x), self.critic_layer(x)

class Value_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Value_Model, self).__init__()   

        self.nn_layer = nn.Sequential(
                nn.Linear(state_dim, 312),
                nn.ReLU(),
                nn.Linear(312, 312),
                nn.ReLU(),
                nn.Linear(312, 1)
              ).float().to(device)
        
    def forward(self, states):
        return self.nn_layer(states)

class PolicyMemory(Dataset):
    def __init__(self):
        self.actions        = [] 
        self.states         = []
        self.rewards        = []
        self.dones          = []     
        self.next_states    = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32), np.array(self.actions[idx], dtype = np.float32), \
            np.array([self.rewards[idx]], dtype = np.float32), np.array([self.dones[idx]], dtype = np.float32), np.array(self.next_states[idx], dtype = np.float32)      

    def get_all(self):
        return self.states, self.actions, self.rewards, self.dones, self.next_states        
    
    def save_all(self, states, actions, rewards, dones, next_states):
        self.actions = self.actions + actions
        self.states = self.states + states
        self.rewards = self.rewards + rewards
        self.dones = self.dones + dones
        self.next_states = self.next_states + next_states
    
    def save_eps(self, state, action, reward, done, next_state):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]  

class AuxMemory(Dataset):
    def __init__(self):
        self.states = []

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype = np.float32)

    def save_all(self, states):
        self.states = self.states + states

    def clear_memory(self):
        del self.states[:]

class Continous():
    def sample(self, mean, std):
        distribution    = Normal(mean, std)
        return distribution.sample().float().to(device)
        
    def entropy(self, mean, std):
        distribution    = Normal(mean, std)    
        return distribution.entropy().float().to(device)
      
    def logprob(self, mean, std, value_data):
        distribution    = Normal(mean, std)
        return distribution.log_prob(value_data).float().to(device)

    def kl_divergence(self, mean1, std1, mean2, std2):
        distribution1   = Normal(mean1, std1)
        distribution2   = Normal(mean2, std2)

        return kl_divergence(distribution1, distribution2).float().to(device)  

class PolicyFunction():
    def __init__(self, gamma = 0.99, lam = 0.95):
        self.gamma  = gamma
        self.lam    = lam

    def monte_carlo_discounted(self, rewards, dones):
        running_add = 0
        returns     = []        
        
        for step in reversed(range(len(rewards))):
            running_add = rewards[step] + (1.0 - dones[step]) * self.gamma * running_add
            returns.insert(0, running_add)
            
        return torch.stack(returns)
      
    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1 - done) * self.gamma * next_value           
        return q_values
      
    def generalized_advantage_estimation(self, values, rewards, next_values, dones):
        gae     = 0
        adv     = []     

        delta   = rewards + (1.0 - dones) * self.gamma * next_values - values          
        for step in reversed(range(len(rewards))):
            gae = delta[step] + (1.0 - dones[step]) * self.gamma * self.lam * gae
            adv.insert(0, gae)
            
        return torch.stack(adv)

class TrulyPPO():
    def __init__(self, policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma, lam):
        self.policy_kl_range    = policy_kl_range
        self.policy_params      = policy_params
        self.value_clip         = value_clip
        self.vf_loss_coef       = vf_loss_coef
        self.entropy_coef       = entropy_coef

        self.distributions      = Continous()
        self.policy_function    = PolicyFunction(gamma, lam)

    def compute_loss(self, action_mean, action_std, old_action_mean, old_action_std, values, old_values, next_values, actions, rewards, dones):    
        # Don't use old value in backpropagation
        Old_values          = old_values.detach()
        Old_action_mean     = old_action_mean.detach()

        # Getting general advantages estimator and returns
        Advantages      = self.policy_function.generalized_advantage_estimation(values, rewards, next_values, dones)
        Returns         = (Advantages + values).detach()
        Advantages      = ((Advantages - Advantages.mean()) / (Advantages.std() + 1e-6)).detach() 

        # Finding the ratio (pi_theta / pi_theta__old):      
        logprobs        = self.distributions.logprob(action_mean, action_std, actions)
        Old_logprobs    = self.distributions.logprob(Old_action_mean, old_action_std, actions).detach() 

        # Finding Surrogate Loss
        ratios          = (logprobs - Old_logprobs).exp() # ratios = old_logprobs / logprobs        
        Kl              = self.distributions.kl_divergence(Old_action_mean, old_action_std, action_mean, action_std)

        pg_targets  = torch.where(
            (Kl >= self.policy_kl_range) & (ratios > 1),
            ratios * Advantages - self.policy_params * Kl,
            ratios * Advantages
        )
        pg_loss     = pg_targets.mean()

        # Getting entropy from the action probability 
        dist_entropy    = self.distributions.entropy(action_mean, action_std).mean()

        # Getting Critic loss by using Clipped critic value
        if self.value_clip is None:
            critic_loss   = ((Returns - values).pow(2) * 0.5).mean()
        else:
            vpredclipped  = old_values + torch.clamp(values - Old_values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
            vf_losses1    = (Returns - values).pow(2) * 0.5 # Mean Squared Error
            vf_losses2    = (Returns - vpredclipped).pow(2) * 0.5 # Mean Squared Error        
            critic_loss   = torch.max(vf_losses1, vf_losses2).mean()                

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss 
        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss

class JointAux():
    def __init__(self):
        self.distributions  = Continous()

    def compute_loss(self, action_mean, action_std, old_action_mean, old_action_std, values, Returns):
        # Don't use old value in backpropagation
        Old_action_mean     = old_action_mean.detach()

        # Finding KL Divergence                
        Kl              = self.distributions.kl_divergence(Old_action_mean, old_action_std, action_mean, action_std).mean()
        aux_loss        = ((Returns - values).pow(2) * 0.5).mean()

        return aux_loss + Kl

class Agent():  
    def __init__(self, state_dim, action_dim, is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                 batchsize, PPO_epochs, gamma, lam, learning_rate):        
        self.policy_kl_range    = policy_kl_range 
        self.policy_params      = policy_params
        self.value_clip         = value_clip    
        self.entropy_coef       = entropy_coef
        self.vf_loss_coef       = vf_loss_coef
        self.batchsize          = batchsize
        self.PPO_epochs         = PPO_epochs
        self.is_training_mode   = is_training_mode
        self.action_dim         = action_dim
        self.std                = torch.ones([1, action_dim]).float().to(device)

        self.policy             = Policy_Model(state_dim, action_dim)
        self.policy_old         = Policy_Model(state_dim, action_dim)
        self.policy_optimizer   = Adam(self.policy.parameters(), lr = learning_rate)

        self.value              = Value_Model(state_dim, action_dim)
        self.value_old          = Value_Model(state_dim, action_dim)
        self.value_optimizer    = Adam(self.value.parameters(), lr = learning_rate)

        self.policy_memory      = PolicyMemory()
        self.policy_loss        = TrulyPPO(policy_kl_range, policy_params, value_clip, vf_loss_coef, entropy_coef, gamma, lam)

        self.aux_memory         = AuxMemory()
        self.aux_loss           = JointAux()
         
        self.distributions      = Continous()

        if is_training_mode:
          self.policy.train()
          self.value.train()
        else:
          self.policy.eval()
          self.value.eval()

    def save_eps(self, state, action, reward, done, next_state):
        self.policy_memory.save_eps(state, action, reward, done, next_state)

    def act(self, state):
        state           = torch.FloatTensor(state).unsqueeze(0).to(device).detach()
        action_mean, _  = self.policy(state)
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action  = self.distributions.sample(action_mean, self.std)
        else:
            action  = action_mean  
              
        return action.squeeze(0).cpu().numpy()

    # Get loss and Do backpropagation
    def training_ppo(self, states, actions, rewards, dones, next_states):
        action_mean, _      = self.policy(states)
        values              = self.value(states)
        old_action_mean, _  = self.policy_old(states)
        old_values          = self.value_old(states)
        next_values         = self.value(next_states)

        loss                = self.policy_loss.compute_loss(action_mean, self.std, old_action_mean, self.std, values, old_values, next_values, actions, rewards, dones)

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        loss.backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

    def training_aux(self, states):
        Returns                         = self.value(states).detach()

        action_mean, values             = self.policy(states)
        old_action_mean, _              = self.policy_old(states)

        joint_loss                      = self.aux_loss.compute_loss(action_mean, self.std, old_action_mean, self.std, values, Returns)

        self.policy_optimizer.zero_grad()
        joint_loss.backward()
        self.policy_optimizer.step()

    # Update the model
    def update_ppo(self):
        dataloader  = DataLoader(self.policy_memory, self.batchsize, shuffle = False)#original is False

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):
            for states, actions, rewards, dones, next_states in dataloader:
                self.training_ppo(states.float().to(device), actions.float().to(device), rewards.float().to(device), dones.float().to(device), next_states.float().to(device))

        # Clear the memory
        states, _, _, _, _ = self.policy_memory.get_all()
        self.aux_memory.save_all(states)
        self.policy_memory.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_old.load_state_dict(self.value.state_dict())

    def update_aux(self):
        dataloader  = DataLoader(self.aux_memory, self.batchsize, shuffle = False)#original is False

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for states in dataloader:
                self.training_aux(states.float().to(device))

        # Clear the memory
        self.aux_memory.clear_memory()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save_weights(self):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.policy_optimizer.state_dict()
            }, 'Trials/policy.tar')
        
        torch.save({
            'model_state_dict': self.value.state_dict(),
            'optimizer_state_dict': self.value_optimizer.state_dict()
            }, 'Trials/value.tar')
        
    def load_weights(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        policy_path = base_path + "/Trials/policy.tar"
        policy_checkpoint = torch.load(policy_path)
        self.policy.load_state_dict(policy_checkpoint['model_state_dict'])
        self.policy_optimizer.load_state_dict(policy_checkpoint['optimizer_state_dict'])

        value_path = base_path + "/Trials/value.tar"
        value_checkpoint = torch.load(value_path)
        self.value.load_state_dict(value_checkpoint['model_state_dict'])
        self.value_optimizer.load_state_dict(value_checkpoint['optimizer_state_dict'])

class Runner():
    def __init__(self, env, agent, render, training_mode, n_update, n_aux_update, max_action):
        self.env = env
        self.agent = agent
        self.render = render
        self.training_mode = training_mode
        self.n_update = n_update
        self.n_aux_update = n_aux_update
        self.max_action = max_action

        self.t_updates = 0
        self.t_aux_updates = 0

    def run_episode(self):
        ############################################
        state = self.env.reset(test=False)    
        done = False
        total_reward = 0
        eps_time = 0
        ep_true = 0
        arr_true = []
        #arr_time = []
        arr_reward = []
        total_eps = 0
        ############################################ 
        for _ in range(1, 1501):#or while True:
            if eps_time==1501 :
                break
            action = self.agent.act(state) 

            action_gym = np.clip(action, -1.0, 1.0) * self.max_action
            next_state, reward, true_reward, done = self.env.step(action_gym)

            ep_true += true_reward
            eps_time += 1 
            self.t_updates += 1
            total_reward += reward
            
            if self.training_mode:
                #self.agent.policy_memory.save_eps(state.tolist(), action.tolist(), reward, float(done), next_state.tolist()) 
                self.agent.policy_memory.save_eps(state, action, reward, float(done), next_state) 
                
            state = next_state
                    
            if self.render:
                self.env.render()
            
            if self.training_mode and self.n_update is not None and self.t_updates == self.n_update:
                print("In the trajectory loop\n")
                self.agent.update_ppo()
                self.t_updates = 0
                self.t_aux_updates += 1

                if self.t_aux_updates == self.n_aux_update:
                    self.agent.update_aux()
                    self.t_aux_updates = 0

            if done:
                total_eps += 1
                arr_reward.append(total_reward)
                #arr_time.append(eps_time)
                arr_true.append(ep_true)

                ep_true = 0
                total_reward = 0
                state = self.env.reset(test=False)  

                #break                
        
        if self.training_mode and self.n_update is None:
            print("In the n_update is None part\n")
            self.agent.update_ppo()
            self.t_aux_updates += 1

            if self.t_aux_updates == self.n_aux_update:
                self.agent.update_aux()
                self.t_aux_updates = 0
                    
        return np.mean(arr_reward), eps_time, np.mean(arr_true), total_eps

def plot(datas):
    print('----------')

    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()

    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))

def main(model_file,loading, load_after_iters):
    ############## Hyperparameters ##############
    load_weights        = loading # If you want to load the agent, set this to True
    save_weights        = False # If you want to save the agent, set this to True
    training_mode       = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold    = 300 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
    using_google_drive  = False

    render              = False # If you want to display the image, set this to True. Turn this off if you run this in Google Collab
    n_update            = 1500 # How many episode before you update the Policy. Recommended set to 128 for Discrete
    n_plot_batch        = 100000000 # How many episode you want to plot the result
    n_episode           = 825 # How many episode you want to run
    n_saved             = 5 # How many episode to run before saving the weights

    policy_kl_range     = 0.03 # Set to 0.0008 for Discrete
    policy_params       = 5 # Set to 20 for Discrete
    value_clip          = 2.0 # How many value will be clipped. Recommended set to the highest or lowest possible reward
    entropy_coef        = 0.01 # How much randomness of action you will get
    vf_loss_coef        = 1.0 # Just set to 1
    batchsize           = 512 # How many batch per update. size of batch = n_update / batchsize. Rocommended set to 4 for Discrete
    PPO_epochs          = 4 # How many epoch per update
    n_aux_update        = 8
    max_action          = 1.0
    
    gamma               = 0.99 # Just set to 0.99
    lam                 = 0.95 # Just set to 0.95
    learning_rate       = 1e-5 # Just set to 0.95
    total_timesteps     = 0
    total_episodes      = 0
    ############################################# 
    #env_name            = 'Pendulum-v0' # Set the env you want
    #env                 = gym.make(env_name)
    #model_file          = "../../../osim-rl/osim/models/healthy_Leanne.osim"
    env                 = ProstheticsEnvMulticlip(visualize=False, model_file=model_file, integrator_accuracy=1e-2)

    state_dim           = env.observation_space.shape[0]
    action_dim          = env.action_space.shape[0]

    agent               = Agent(state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                            batchsize, PPO_epochs, gamma, lam, learning_rate)  

    runner              = Runner(env, agent, render, training_mode, n_update, n_aux_update, max_action)
    #############################################     

    save_prefix = model_file[10:-5]
    i_episode = 1
    if using_google_drive:
        from google.colab import drive
        drive.mount('/test')

    if load_weights == 1:
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_f = os.path.normpath(
            base_path + "/../../../../" + save_prefix + '/models/' + save_prefix + "_afterIter_" + str(
                load_after_iters))    
        agent.load_weights()
        # Restore the variables from file
        data = genfromtxt(save_prefix + '/test_afterIter_' + str(load_after_iters) + '.csv', delimiter=',')
        num_rows = sum(1 for row in data)
        for i in range(len(data)):
            data_vector = data[i]
            total_episodes = int(data_vector[0])
            total_timesteps = int(data_vector[1])
            i_episode = int(data_vector[2])

    """if load_weights: #Add the true rews, total episodes and timesteps
                    agent.load_weights()
                    print('Weights Loaded')
                    i_episode = load_after_iters"""


    start = time.time()

    try:
        while True: #for i_episode in range(1, n_episode + 1):

            if i_episode==n_episode+1 :
                break

            iter_time = time.time()
            total_reward, eps_time, ep_true, run_eps = runner.run_episode()
            total_timesteps += eps_time
            total_episodes += run_eps
            print('Iteration {} \t episodes {} \t t_reward: {} \t time: {} \t true: {} \t real time {}'.format(i_episode, total_episodes, total_reward, total_timesteps, ep_true, time.time()-iter_time))
            
            i_episode += 1
            if i_episode%1==0 or i_episode==1:
                base_path = os.path.dirname(os.path.abspath(__file__))
                f = open(base_path + "/../../../../" + save_prefix + "/trues_ppg.txt", "a+")
                f.write("Episodes %d    " % total_episodes)
                f.write("Reward  %d\r\n" % ep_true)
                f.close()
                l = open(base_path + "/../../../../" + save_prefix + "/iterations.txt", "a+")
                l.write("%d\r\n" % i_episode)
                l.close()
                m = open(base_path + "/../../../../" + save_prefix + "/timesteps.txt", "a+")
                m.write("%d\r\n" % total_timesteps)
                m.close()
                agent.save_weights()
                base_path = os.path.dirname(os.path.abspath(__file__))
                model_f = os.path.normpath(
                    base_path + '/../../../' + save_prefix + '/models/' + save_prefix + "_afterIter_" + str(
                        i_episode))
                if total_episodes < 100:
                    size = total_episodes
                else:
                    size = 100
                asd = np.zeros((size, 4), dtype=np.int32)
                for i in range(size):
                    asd[i] = [total_episodes, total_timesteps, i_episode, ep_true]
                    np.savetxt(save_prefix + '/test_afterIter_' + str(i_episode) + '.csv', asd, delimiter=",")
            ##############################################################

    except KeyboardInterrupt:        
        print('\nTraining has been Shutdown \n')

    finally:
        finish = time.time()
        timedelta = finish - start
        print('Timelength: {}'.format(str( datetime.timedelta(seconds = timedelta) )))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('restore', type=int, default=1)
    parser.add_argument('load_iters', type=int, default=1)
    parser.add_argument('env', type=str, default='HalfCheetah-v2')
    args = parser.parse_args()
    load_iters = args.load_iters

    if args.load_iters == 1:
        with open(args.env[10:-5] + '/iterations.txt', 'r') as f:
            lines = f.read().splitlines()
            last_iter = int(lines[-1])
            load_iters = last_iter
    main(args.env, args.restore, load_iters)