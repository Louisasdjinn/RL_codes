import argparse
import os
import pickle
import time
from collections import namedtuple
from itertools import count
# edit from: https://github.com/Unity-Technologies/ml-agents/blob/master/notebooks/getting-started.ipynb
#env_name = "../envs/GridWorld"  # Name of the Unity environment binary to launch
env_name = None  # Directly interact with editor
train_mode = True  # Whether to run the environment in training or inference mode
#1. Set environment parameters
import matplotlib.pyplot as plt
import numpy as np
import sys

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel

#%matplotlib inline

print("Python version:")
print(sys.version)

    # check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

engine_configuration_channel = EngineConfigurationChannel()
env = UnityEnvironment(base_port = 5004, file_name=env_name, side_channels = [engine_configuration_channel])
    # 2. Load dependencies


    #Reset the environment
env.reset()

    # Set the default brain to work with
group_name = env.get_agent_groups()[0]
group_spec = env.get_agent_group_spec(group_name)

    # Set the time scale of the engine
engine_configuration_channel.set_configuration_parameters(time_scale = 3.0)

    # 3. Start the environment

    # Get the state of the agents
step_result = env.get_step_result(group_name)

    # Examine the number of observations per Agent
print("Number of observations : ", len(group_spec.observation_shapes))

    # Examine the state space for the first observation for all agents
print("Agent state looks like: \n{}".format(step_result.obs[0]))

    # Examine the state space for the first observation for the first agent
print("Agent state looks like: \n{}".format(step_result.obs[0][0]))
print("Agent state looks like: \n{}".format(step_result.obs[0][0].shape))
print("Action_size is: \n{}".format(group_spec.discrete_action_branches))
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
            print("action",action)
        env.set_actions(group_name, action)
        env.step()
        #print("env.step",env.step)
        step_result = env.get_step_result(group_name)
        episode_rewards += step_result.reward[0]
        done = step_result.done[0]
    
    print("Total reward this episode: {}".format(episode_rewards))

env.close()
    # 6. Close the environment when finished

# 5. Take random actions in the environment
def main():
    {
    }

if __name__ == '__main__':
    main()

