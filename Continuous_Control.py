#!/usr/bin/env python
# coding: utf-8

# # Continuous Control
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the second project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.
# 
# ### 1. Start the Environment
# 
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# In[1]:
'''
To convert an .ipynb file to an .py file, run the following command.

$ ls
Continuous_Control.ipynb
$ jupyter nbconvert --to script Continuous_Control.ipynb
[NbConvertApp] Converting notebook Continuous_Control.ipynb to script
[NbConvertApp] Writing 10294 bytes to Continuous_Control.py
$ ls
Continuous_Control.ipynb  Continuous_Control.py
$
For details, refer to https://github.com/aimldl/computing_environments/blob/master/jupyter/convert_notebook_to_script.md
'''

from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import pandas as pd
import random
import matplotlib.pyplot as plt
import torch
from ddpg_agent import Agent
get_ipython().run_line_magic('matplotlib', 'inline')


# Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.
# 
# - **Mac**: `"path/to/Reacher.app"`
# - **Windows** (x86): `"path/to/Reacher_Windows_x86/Reacher.exe"`
# - **Windows** (x86_64): `"path/to/Reacher_Windows_x86_64/Reacher.exe"`
# - **Linux** (x86): `"path/to/Reacher_Linux/Reacher.x86"`
# - **Linux** (x86_64): `"path/to/Reacher_Linux/Reacher.x86_64"`
# - **Linux** (x86, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86"`
# - **Linux** (x86_64, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86_64"`
# 
# For instance, if you are using a Mac, then you downloaded `Reacher.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
# ```
# env = UnityEnvironment(file_name="Reacher.app")
# ```

# In[2]:


get_ipython().system('pwd')
#env = UnityEnvironment(file_name='unity_env/Reacher_Linux_NoVis/Reacher.x86_64',no_graphics=True)
env = UnityEnvironment(file_name='unity_env/multi_agents/Reacher_Linux_NoVis/Reacher.x86_64')


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[3]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
# 
# The observation space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm.  Each action is a vector with four numbers, corresponding to torque applicable to two joints.  Every entry in the action vector must be a number between `-1` and `1`.
# 
# Run the code cell below to print some information about the environment.

# In[4]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# ### 3. Take Random Actions in the Environment
# 
# In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
# 
# Once this cell is executed, you will watch the agent's performance, if it selects an action at random with each time step.  A window should pop up that allows you to observe the agent, as it moves through the environment.  
# 
# Of course, as part of the project, you'll have to change the code so that the agent is able to use its experience to gradually choose better actions when interacting with the environment!

# In[5]:


'''
env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
    actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
'''


# When finished, you can close the environment.

# In[6]:


#env.close()


# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```
# 

# In[7]:


env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)
# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
num_agents=states.shape[0]
print('Number of agent:', num_agents)
print('size of states for each agent:', state_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:{}".format(device))


# In this section I will use DDPG to train the actor/critic network. First I will utilize 1 agents then try 20 agents. Updates were doen every timestep. 

# In[ ]:





# In[8]:



def train_agent(n_episodes=500, max_t=1000, print_every=10):
 scores_deque = deque(maxlen=100)
 episode_scores = []                                     #average scores of agents per episode
 episode_mean_scores=[]                                  #average scores of agents for last 100 episode 
 last_mean_max=0   

 for i_episode in range(1, n_episodes+1):      
     env_info = env.reset(train_mode=True)[brain_name]                 # reset the environment    
     states = env_info.vector_observations   
     scores = np.zeros(num_agents)
     agent.reset()
     #for agent in agents:
     #    agent.reset()
     
     #print(i_episode)
     for t in range(max_t):
         actions =agent.act(states)
         print("actions from agents:{}".format(actions))
         #np.array([agents[i].act(states[i]) for i in range(num_agents)])
         #actions = np.clip(actions, -1, 1) 
         #print("actions:{}".format(actions))                            # all actions between -1 and 1
         env_info = env.step(actions)[brain_name]                        # send all actions to tne environment
         next_states = env_info.vector_observations                      # get next state (for each agent)
         rewards = env_info.rewards
         dones = env_info.local_done  
         # get reward (for each agent)
         #print(rewards,i_episode,t)
         
         #print("passed env_info")                                        
         
         #for i in range(num_agents):                                    #update network 
         #    agents[i].step(t, states[i], actions[i], rewards[i], next_states[i], dones[i]) 
         agent.step(t,states, actions, rewards, next_states, dones)
         states = next_states
         scores += rewards
         #print(scores)
         #print("t:{}".format(t))
         if np.any(dones):
             break 
     score=np.mean(scores)
     scores_deque.append(score)
     episode_scores.append(score)
     
     episode_mean_scores.append(np.mean(scores_deque)) 
     
     print('\rEpisode {} \tScore: {:.2f} \t Avg. Score: {:.2f}'.format(i_episode,score, np.mean(scores_deque)), end="")
     
     if np.mean(scores_deque)>=30.0:
         print('\nEnvironment solved in {:d} episodes!'.format(i_episode-100, np.mean(scores_deque)))
         if(np.mean(scores_deque)>last_mean_max):
             torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
             torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
             last_mean_max=np.mean(scores_deque)
     if i_episode % print_every == 0:
        print('\rEpisode {} \tScore: {:.2f} \t Avg. Score: {:.2f}'.format(i_episode,score, np.mean(scores_deque)))
         
 return episode_scores, episode_mean_scores


# In[9]:


#agents=[]
#for i in range(num_agents):
#    agent = Agent(state_size=state_size, action_size=action_size, random_seed=10)
#    agents.append(agent)
agent=Agent(num_agents=num_agents, state_size=state_size, action_size=action_size, random_seed=10)
scores, mean_scores = train_agent()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores,label='20Agents_update_every_timestep' )
plt.ylabel('Score')
plt.plot(np.arange(len(scores)), mean_scores, c='r', label='Mean Avearge Score')
plt.xlabel('Episode #')
plt.legend(loc='upper left')
plt.show()


'''
#Train Agent with DQN agent
agent = Agent(state_size, action_size, seed=0, use_duel_dqn=False, use_double_dqn=False,use_basic=True)
scores, mean_scores,episode_achieved = train_agent(agent)
condition.append([0, 0,'FC64'])
score_history.append(max(scores))
mean_score_history.append(max(mean_scores))
first_episode_achieved.append(episode_achieved)
'''


# In[ ]:




