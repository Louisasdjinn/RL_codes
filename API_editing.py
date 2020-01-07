import argparse
import os
import pickle
import time
import torch
from collections import namedtuple
from itertools import count
# edit from: https://github.com/Unity-Technologies/ml-agents/blob/master/notebooks/getting-started.ipynb

#1. Set environment parameters
import matplotlib.pyplot as plt
import numpy as np
import sys

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
import DQN_Prioritized_EXP as DQN
#from DQN import DQN, Transition

#env_name = "../envs/GridWorld"  # Name of the Unity environment binary to launch
env_name = '/home/louis/Desktop/u/push.x86_64'  # Directly interact with editor
train_mode = True  # Whether to run the environment in training or inference mode
# Hyper-parameters
seed = 1
num_episodes = 200000000
#env = gym.make('CartPole-v0').unwrapped
torch.manual_seed(seed)

datalist=[0,0,0] #refresh each episode to save the data


#%matplotlib inline
print("Python version:")
print(sys.version)
# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

engine_configuration_channel = EngineConfigurationChannel()
env = UnityEnvironment(base_port = 5004, worker_id = 4,file_name=env_name, side_channels = [engine_configuration_channel])
# 2. Load dependencies
#Reset the environment
env.reset()
# Set the default brain to work with
group_name = env.get_agent_groups()[0]
group_spec = env.get_agent_group_spec(group_name)
# Set the time scale of the engine
engine_configuration_channel.set_configuration_parameters(width=800,height=600)

# 3. Start the environment
# Get the state of the agents
step_result = env.get_step_result(group_name)
# Examine the number of observations per Agent
print("Number of observations : ", len(group_spec.observation_shapes))
# Examine the state space for the first observation for all agents
print("Agent state looks like: \n{}".format(step_result.obs[0]))
# Examine the state space for the first observation for the first agent
print("Agent state looks like: \n{}".format(step_result.obs[0][0]))
# Is there a visual observation ?
vis_obs = any([len(shape) == 3 for shape in group_spec.observation_shapes])
print("Is there a visual observation ?", vis_obs)
# Examine the visual observations
if vis_obs:
    vis_obs_index = next(i for i,v in enumerate(group_spec.observation_shapes) if len(v) == 3)
    print("Agent visual observation look like:")
    obs = step_result.obs[vis_obs_index]
    plt.imshow(obs[0,:,:,:])
# 4. Examine the observation and state spaces

def CloseEnv():
    env.close()
    # 6. Close the environment when finished

def writedata(datalist):
    # python list that is needed to be save in the csv or txt file
    # convert list to array
    data_array = np.array(datalist)
    # saving...
    np.savetxt('data_exp.txt',data_array,fmt = '%.7f',delimiter=',')
    print ('Finish saving txt')
    #save the current data

def plotdata():
    data = np.loadtxt('data_exp.txt',dtype = None,delimiter=',')
    plt.plot(data[:,0],data[:,1],data[:,2])
    plt.show()
    #draw the results of data with matplotb

beta_start = 0.4
beta_frames = 1000 
beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

num_frames = 1000000000
batch_size = 256
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

env.reset()
#agent.step = 0#TODO
step_result = env.get_step_result(group_name)
done = False
reward = 0
state = step_result.obs[0]    

for frame_idx in range(1, num_frames + 1):
    epsilon = DQN.epsilon_by_frame(frame_idx)
    action_value = DQN.current_model.act(state, epsilon)
    action = np.array([[action_value]])
    #action = action_value[np.newaxis,:]
    env.step()
    #print("state1",step_result.obs[0])
    #print(action)
    env.set_actions(group_name, action)# step +1 here
    #print(action)
    #print("state2",step_result.obs[0])
    next_state = step_result.obs[0]
    #print("state3",step_result.obs[0])
    reward += step_result.reward[0]
    done = step_result.done[0]

    DQN.replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done or DQN.replay_buffer.step >= 1000:#TODO
        print(DQN.replay_buffer.step)
        env.reset()
        state = step_result.obs[0] 
        all_rewards.append(episode_reward)
        episode_reward = 0
        DQN.replay_buffer.step = 0
        
    if len(DQN.replay_buffer) > batch_size:
        beta = beta_by_frame(frame_idx)
        loss = DQN.compute_td_loss(batch_size, beta)
        losses.append(loss.data.item())
        
    if frame_idx % 6000000 == 0:
        DQN.plot(frame_idx, all_rewards, losses)
        
    if frame_idx % 1000 == 0:
        DQN.update_target(DQN.current_model, DQN.target_model)
'''
def Iteration():
    loss = []

    agent = DQN()
    for i_ep in range(num_episodes):
        env.reset()
        agent.step = 0
        step_result = env.get_step_result(group_name)
        done = False
        reward = 0
        state = step_result.obs[0]
        #print("step_result.obs[0]",step_result.obs[0])
        #if render: env.render()
        #print("2")
        for agent.step in range(1000):
            action_value = agent.select_action(state)
            action = np.array([[action_value]])
            #action = action_value[np.newaxis,:]
            env.step()
            #print("state1",step_result.obs[0])
            #print(action)
            env.set_actions(group_name, action)# step +1 here
            #print(action)
            #print("state2",step_result.obs[0])
            next_state = step_result.obs[0]
            #print("state3",step_result.obs[0])
            reward += step_result.reward[0]
            done = step_result.done[0]
            #ID_agent = #TODO
            #next_state, reward, done, info = env.step(action)
            #if render: env.render()#draw scene for each timestep
            transition = Transition(state, action, reward, next_state, done)
            agent.store_transition(transition)
            state = next_state
            step_result = env.get_step_result(group_name)
            print('steps:',agent.step)
            agent.update()

            #recording process
            if done or agent.step >= 1000:
                #agent.writer.add_scalar('live/finish_step', agent.step+1, global_step=i_ep)
                global datalist
                data_episode=[i_ep+1,reward,agent.loss_value]
                datalist = np.vstack((datalist,data_episode))
                writedata(datalist)
                #if i_ep % 10 == 0:
                print("episodes {}, step is {} ".format(i_ep, agent.step))
                print("Total reward this episode: {}".format(reward))
                break
'''

'''
def RandomAction():
        for episode in range(10):
            env.reset()
            step_result = env.get_step_result(group_name)
            done = False
            episode_rewards = 0
            while not done:
                
                action_size = group_spec.action_size
                if group_spec.is_action_continuous():
                    action = np.random.randn(step_result.n_agents(), group_spec.action_size)
            
                if group_spec.is_action_discrete():
                    branch_size = group_spec.discrete_action_branches
                    action = np.column_stack([np.random.randint(0, branch_size[i], size=(step_result.n_agents())) for i in range(len(branch_size))])
                
                env.set_actions(group_name, action)
                env.step()
                step_result = env.get_step_result(group_name)
                episode_rewards += step_result.reward[0]
                done = step_result.done[0]
            print("Total reward this episode: {}".format(episode_rewards))
            # 5. Take random actions in the environment
'''

if __name__ == '__main__':
    main()
