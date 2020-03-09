[//]: # "Image References"

[image1]: https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif "Trained Agent"

# Project 2: Build an RL agent that collects Bananas

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.  Unity Machine Learning Agents (ML-Agents) plugin will be used to serve as environments for training intelligent agents. You can read more about ML-Agents by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents)



![Trained Agent][image1]



### Environment 

In this environment, a double-jointed arm can move to target  locations. A reward of +0.1 is provided for each step that the agent's  hand is in the goal location. Thus, the goal of your agent is to  maintain its position at the target location for as many time steps as  possible.

The observation space consists of 33 variables corresponding to  position, rotation, velocity, and angular velocities of the arm. Each  action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number  between -1 and 1.



### Distributed Training 

For this project, we will provide you with two separate versions of the Unity environment:

- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Goal: Solving the Environment

Note that your project submission need only solve one of the two versions of the environment. 

##### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your  agent must get an average score of +30 over 100 consecutive episodes.

##### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is  slightly different, to take into account the presence of many agents.   In particular, your agents must get an average score of +30 (over 100  consecutive episodes, and over all agents).  Specifically,

- After each episode, we add up the rewards that each agent received  (without discounting), to get a score for each agent.  This yields 20  (potentially different) scores.  We then take the average of these 20  scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

As an example, consider the plot below, where we have plotted the **average score** (over all 20 agents) obtained with each episode.



[![Plot of average scores (over all agents) with each episode.](https://video.udacity-data.com/topher/2018/July/5b48f845_unknown/unknown.png)Plot of average scores (over all agents) with each episode. ](https://classroom.udacity.com/nanodegrees/nd893/parts/286e7d2c-e00c-4146-a5f2-a490e0f23eda/modules/089d6d51-cae8-4d4b-84c6-9bbe58b8b869/lessons/5b822b1d-5c89-4fd5-9b52-a02ddcfd3385/concepts/540e7c18-6831-417e-981c-309c3ffedd56#)



The environment is considered solved, when the average (over 100 episodes) of those **average scores** is at least +30.  In the case of the plot above, the environment was solved at episode 63, since the average of the **average scores** from episodes 64 to 163 (inclusive) was greater than +30.

### Getting Started

#### Install packages and dependencies

For the Training, I used a p2.xlarge type AWS EC2 instance (Ubuntu based Deep Learning AMI, AMI ID i-0aeca4a4e5d610469) the Seoul Region, where I closely located.  Most of the utilities are already installed  in Deep learning AMI so minor correction was made on requirment.txt file. 



1. Create and activate a new conda environment 

   ###### Linux or Mac:		

   ```
   conda create --name drlnd python=3.6
   source activate drlnd
   ```

   ###### Windows:

   ```
   conda create --name drlnd python=3.6
   activate drlnd
   ```

2. Install OpenAI gym in the environment

   ```
   pip install gym
   ```

3. Clone following repository and install additional dependencies.

   ```
   git clone https://github.com/jihys/p2_continuous-control.git
   #Alterntively you can download original codes from udcity github. 
   #git clone https://github.com/udacity/deep-reinforcement-learning.git`
   cd python`
   vi requirement.txt` 
   #"Correct requirement.txt as needed"`
   pip install .
   ```

   Note: Please comment out jupyter,ipykernel in the "deep-reinforcement-learning/python/requirment.txt" file and install required packages. 

   ```
   tensorflow==1.7.1
   Pillow>=4.2.1
   matplotlib
   numpy>=1.11.0
   #jupyter
   pytest>=3.2.2
   docopt
   pyyaml
   protobuf==3.5.2
   grpcio==1.11.0
   torch>=0.4.0
   pandas
   scipy
   #ipykernel
   ```


#### Unity Environment Setup 

For this project, you will **not** need to install Unity - this is because we have already built the environment for you, and  you can download it from one of the links below.  You need only select  the environment that matches your operating system:

##### Version 1: One (1) Agent

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

##### Version 2: Twenty (20) Agents

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Then, place the file in the `p2_continuous-control/` folder and unzip (or decompress) the file.

(*For Windows users*) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit  version or 64-bit version of the Windows operating system.

(*For AWS*) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (*To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.*)

 

### Instructions

Follow the instructions in `Continuous_control.ipynb` to get started with training your own agent. `Model.py` contains the Actor, Critic networks while  `ddpg_agents.py ` includes codes for replay buffer and Agent network update and training methods. 

 

