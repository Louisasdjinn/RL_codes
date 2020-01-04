import argparse
import os
import pickle
import time
from collections import namedtuple, deque
from itertools import count
import math, random
import gym

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from IPython.display import clear_output

#from torch.distributions import Categorical, Normal
#from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# Call from source env
#from mlagents_envs.environment import UnityEnvironment
#from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel

#Use Cuda
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

# Hyper-parameters
seed = 1
num_episodes = 2000
#Env Parameters
'''
env_name = None  # Directly interact with editor
engine_configuration_channel = EngineConfigurationChannel()
env = UnityEnvironment(base_port = 5004, file_name=env_name, side_channels = [engine_configuration_channel])
group_name = env.get_agent_groups()[0]
group_spec = env.get_agent_group_spec(group_name)
step_result = env.get_step_result(group_name)
'''
num_state = 105
#num_state = step_result.obs[0][0].shape[0]
num_action = 7
#num_action = group_spec.discrete_action_branches[0]
torch.manual_seed(seed)
#env.seed(seed)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state','done'])

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_state, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        action_value = self.fc5(x)
        return action_value

class DQN():

    capacity = 1000*600 # start record exp after 100 episodes
    learning_rate = 1e-3
    memory_count = 0
    batch_size = 256
    gamma = 0.995
    update_count = 0
    step = 0


    def __init__(self):
        super(DQN, self).__init__()
        self.target_net, self.act_net = Net(), Net()
        self.memory = deque(maxlen = self.capacity)
        # self.memory = [None]*self.capacity
        self.optimizer = optim.Adam(self.act_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./DQN/logs')
        self.loss_value = 0


    def select_action(self,state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)#1x1x105
        value = self.act_net(state)#1x1x7
        #print("value",value)
        action_max_value, index = torch.max(value, 2)#return the max in each line and its index:torch.max(a,1)
        #print("index",index)#1x1
        #print("action_max_value",action_max_value)#1x1
        action = index.item() # only correspond to the Python number from a single value tensor
        self.step += 1
        #print("action",action)
        if np.random.rand(1) >= 0.9: # epslion greedy
            action = np.random.choice(range(num_action), 1).item()
        return action

    def store_transition(self,transition):#TODO transition is called every timestep
        self.memory.append(transition)
        #index = self.memory_count % self.capacity
        #self.memory[index] = transition
        self.memory_count += 1
        return self.memory_count >= self.capacity
    
    def sample(self,batch_size):
        experience = random.sample(self.memory, batch_size)
        states = torch.from_numpy(np.vstack([Transition.state for Transition in experience if Transition is not None])).float().view(batch_size,1,-1)
        actions = torch.from_numpy(np.vstack([Transition.action for Transition in experience if Transition is not None])).long().view(batch_size,-1)
        rewards = torch.from_numpy(np.vstack([Transition.reward for Transition in experience if Transition is not None])).float().view(batch_size)
        next_states = torch.from_numpy(np.vstack([Transition.next_state for Transition in experience if Transition is not None])).float().view(batch_size,1,-1)
        dones = torch.from_numpy(np.vstack([Transition.done for Transition in experience if Transition is not None]).astype(np.uint8)).float().view(batch_size)#set done for cut the relationship between finishment and initiallization   
        #size of done and reward should be related
        return states,actions,next_states,rewards,dones
    
    def update(self):
        if self.memory_count >= self.capacity:
            state,action,next_state,reward,done = self.sample(self.batch_size)
            #print("Start updating")
            #state = torch.tensor([t.state for t in self.memory]).float()#Translate state from 1x1x105 to 100(memory size)x1x105 (Utilize the first dimension of state(unsqueeze))
            #action = torch.LongTensor([t.action for t in self.memory]).view(-1,1).long()
            #reward = torch.tensor([t.reward for t in self.memory]).float()
            #next_state = torch.tensor([t.next_state for t in self.memory]).float()

            #reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            #Dimension Check
            #print("reward",reward)
            #print("shape",reward.shape)
            #print("targetnet",self.target_net(next_state).max(2)[0])# view all elements
            #print("shape",self.target_net(next_state))
            #print("shape",self.target_net(next_state).shape) #100x1x7
            #print("targetnet",self.target_net(next_state).max(2)[0].squeeze(1))
            #print("CheckDim",self.target_net(next_state).max(2)[0].squeeze(1).shape == reward.shape)
            #print("shape",self.act_net(state)) #100x1x7
            with torch.no_grad():
                target_v = reward + (1-done) *(self.gamma * self.target_net(next_state).max(2)[0].squeeze(1))# 100
            #print("target_v",target_v)
            #print("shape",target_v.shape)

            #Update...
            '''
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
                #print("index",index) #1x100
                #print("action",action)
                #print("shape",action.shape) #100x1
                #print("target_v[index].unsqueeze(1)",target_v[index].unsqueeze(1).shape)
            '''
            v = (self.act_net(state).squeeze(1).gather(1, action))#Q value #with the dimensions decreased manually
            loss = self.loss_func(target_v.unsqueeze(1), (self.act_net(state).squeeze(1).gather(1, action)))#Target - Q value of act net
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_value = loss.detach().numpy() # for transfer the value to the API
            self.writer.add_scalar('loss/value_loss', loss, self.update_count)
            #print("loss",loss)
            self.update_count +=1
            if self.update_count % 100 ==0:
                self.target_net.load_state_dict(self.act_net.state_dict())
        else:
            print("Memory Buff is too less")
def main():

    '''
    agent = DQN()
    for i_ep in range(num_episodes):
        state = env.reset()
        print("1")
        if render: env.render()
        print("2")
        for t in range(10000):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            print("3")
            if render: env.render()#draw scene for each timestep
            transition = Transition(state, action, reward, next_state)
            agent.store_transition(transition)
            state = next_state
            if done or t >=9999:
                agent.writer.add_scalar('live/finish_step', t+1, global_step=i_ep)
                agent.update()
                if i_ep % 10 == 0:
                    print("episodes {}, step is {} ".format(i_ep, t))
                break
    '''

if __name__ == '__main__':
    main()
