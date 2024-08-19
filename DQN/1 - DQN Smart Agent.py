import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from pysc2.agents import base_agent
from pysc2.lib import actions, features
from pysc2.env import sc2_env, run_loop

# Functions
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_BARRACKS = 21
_TERRAN_COMMANDCENTER = 18
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_SCV = 45

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]

# Definition of action constants
ACTION_DO_NOTHING = 'donothing' 
ACTION_SELECT_SCV = 'selectscv' 
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

# List of all defined actions
smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_BUILD_MARINE,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK,
]

# Reward values for specific in-game achievements
KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5

# Define DQN Model
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# SC2 Agent using DQN
class SC2DQNAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SC2DQNAgent, self).__init__()
        self.num_actions = len(smart_actions)  # define smart_actions list as in Program 1
        self.state_size = 10  # Define the size of the state space
        self.dqn = DQN(self.state_size, 50, self.num_actions)  # Hidden size of 50 is arbitrary
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.steps_done = 0
        self.epsilon = 0.9

    def step(self, obs):
        super(SC2DQNAgent, self).step(obs)
        state = self.get_state(obs)
        action = self.select_action(state)
        reward = self.compute_reward(obs)
        next_state = self.get_state(obs)  # update this appropriately based on the action taken
        self.memory.push(state, action, next_state, reward)

        self.learn()
        return self.perform_action(obs, action)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon * 0.02 ** (self.steps_done / 1000)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.dqn(torch.from_numpy(state).float()).max(0)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

        # Prepare data for training
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch_next_state if s is not None])
        state_batch = torch.cat(batch_state)
        action_batch = torch.cat(batch_action)
        reward_batch = torch.cat(batch_reward)

        # Compute Q(s_t, a)
        state_action_values = self.dqn(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_values[non_final_mask] = self.dqn(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def perform_action(self, obs, action):
        # Map the action index to actual game action
        return actions.FunctionCall(_NO_OP, [])

if __name__ == "__main__":
    agent = SC2DQNAgent()
    try:
        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.terran),
                     sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=True) as env:
            run_loop.run_loop([agent], env, max_episodes=10)
    except KeyboardInterrupt:
        pass
