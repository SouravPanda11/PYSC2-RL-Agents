import random
import math
import os
import numpy as np
import pandas as pd
import sys
import absl.flags as flags
import logging  # Import the logging module
from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import features, actions
import torch
import torch.nn as nn
import torch.optim as optim

max_episodes = 1000  # Define the maximum episodes variable

# Constants for setting up the logging directory and filename
log_dir = 'DQN_logs'
log_file_name = f'DQN_logs_{max_episodes}.log'
# Check if the log directory exists, if not create it
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
# Setup logging with dynamic filename based in the 'logs' directory
logging.basicConfig(filename=os.path.join(log_dir, log_file_name), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for actions
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

# Constants for features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

# Constants for identification
_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

# Constants for actions queue
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

# Action definitions
ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))
            

# Deep Q-Network (DQN) model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.01, gamma=0.9, epsilon=0.9):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def select_action(self, state, exclude_actions=[]):
        state = torch.tensor([state], dtype=torch.float32)  # Convert state to tensor and add batch dimension
        q_values = self.model(state).detach().numpy().flatten()  # Get Q-values and convert to numpy array
        for action in exclude_actions:
            q_values[action] = -float('inf')  # Set Q-values of excluded actions to negative infinity
        if random.random() < self.epsilon:
            action = q_values.argmax()  # Exploitation: select the action with the max Q-value
        else:
            valid_actions = [a for a in range(self.action_dim) if a not in exclude_actions]
            action = random.choice(valid_actions)  # Exploration: randomly select a valid action
        return action

    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor([state], dtype=torch.float32)  # State tensor with batch dimension
        next_state = torch.tensor([next_state], dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)

        current_q_values = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_state).max(1)[0]
        expected_q_values = reward + self.gamma * next_q_values * (1 - int(done))

        loss = self.criterion(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self, decay_rate):
        self.epsilon *= decay_rate
        self.epsilon = max(self.epsilon, 0.01)  # Maintain a minimum epsilon to ensure some exploration

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        
class Agent(base_agent.BaseAgent):
    def __init__(self):
        super(Agent, self).__init__()
        self.qlearn = DQNAgent(state_dim=12, action_dim=len(smart_actions))  # Update the dimensions appropriately
        self.previous_action = None
        self.previous_state = None
        self.cc_y = None
        self.cc_x = None
        self.move_number = 0
        logging.info("Initializing the DQNAgent")
        
    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]
    
    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]
        return [x, y]
    
    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')
        return (smart_action, x, y)
        
    def step(self, obs):
        super(Agent, self).step(obs)
        logging.info("Processing a new step")
        
        if obs.last():
            reward = obs.reward
            self.qlearn.learn(self.previous_state, self.previous_action, reward, None, done=True)
            self.previous_action = None
            self.previous_state = None
            self.move_number = 0
            return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
        
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

        current_state = self.extract_state(obs, unit_type)

        if self.move_number == 0:
            self.move_number += 1

            if self.previous_action is not None:
                self.qlearn.learn(self.previous_state, self.previous_action, 0, current_state, done=False)
            
            excluded_actions = self.get_excluded_actions(obs, current_state)
            action_id = self.qlearn.select_action(current_state, exclude_actions=excluded_actions)

            self.previous_state = current_state
            self.previous_action = action_id
            
            smart_action, x, y = self.splitAction(action_id)
            return self.perform_action(obs, smart_action, x, y)
        
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
    
    def extract_state(self, obs, unit_type):
        player_relative = obs.observation['feature_minimap'][_PLAYER_RELATIVE]
        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0
        
        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = len(depot_y) // 69

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = len(barracks_y) // 137

        supply_used = obs.observation['player'][3]
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        worker_supply = obs.observation['player'][6]

        supply_free = supply_limit - supply_used

        state = np.zeros(12)
        state[0:4] = [cc_count, supply_depot_count, barracks_count, army_supply]

        hot_squares = np.zeros(4)
        enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()
        for i in range(len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 32))
            x = int(math.ceil((enemy_x[i] + 1) / 32))
            hot_squares[((y - 1) * 2) + (x - 1)] = 1

        state[4:8] = hot_squares[::-1] if not self.base_top_left else hot_squares

        green_squares = np.zeros(4)
        friendly_y, friendly_x = (player_relative == _PLAYER_SELF).nonzero()
        for i in range(len(friendly_y)):
            y = int(math.ceil((friendly_y[i] + 1) / 32))
            x = int(math.ceil((friendly_x[i] + 1) / 32))
            green_squares[((y - 1) * 2) + (x - 1)] = 1

        state[8:12] = green_squares[::-1] if not self.base_top_left else green_squares
        return state

    def get_excluded_actions(self, obs, current_state):
        excluded_actions = []
        
        # Constants for action indices, ensure these are correctly defined in your environment setup
        ACTION_BUILD_SUPPLY_DEPOT = 1
        ACTION_BUILD_BARRACKS = 2
        ACTION_TRAIN_MARINE = 3
        ACTION_ATTACK_START_INDEX = 4  # Assuming ATTACK actions start from index 4

        supply_depot_count = current_state[1]
        barracks_count = current_state[2]
        supply_free = obs.observation['player'][4] - obs.observation['player'][3]  # supply_limit - supply_used
        worker_supply = obs.observation['player'][6]
        army_supply = obs.observation['player'][5]

        # Exclude actions based on game rules
        if supply_depot_count >= 2 or worker_supply == 0:
            excluded_actions.append(ACTION_BUILD_SUPPLY_DEPOT)
        
        if barracks_count >= 2 or worker_supply == 0:
            excluded_actions.append(ACTION_BUILD_BARRACKS)
        
        if supply_free == 0 or barracks_count == 0:
            excluded_actions.append(ACTION_TRAIN_MARINE)
        
        if army_supply == 0:  # Excluding all attack actions if no army is available
            for attack_index in range(ACTION_ATTACK_START_INDEX, len(smart_actions)):
                excluded_actions.append(attack_index)

        return excluded_actions


    def perform_action(self, obs, smart_action, x, y):
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

        if smart_action == 'buildsupplydepot':
            if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, [x, y]])

        elif smart_action == 'buildbarracks':
            if _BUILD_BARRACKS in obs.observation['available_actions']:
                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, [x, y]])

        elif smart_action == 'buildmarine':
            if _TRAIN_MARINE in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

        elif smart_action.startswith('attack'):
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        return actions.FunctionCall(_NO_OP, [])

def main():
    # max_episodes = 100000
    flags.FLAGS(sys.argv)

    try:
        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True
            ),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=True
        ) as env:
            agent = Agent()
            run_loop.run_loop([agent], env, max_episodes)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        agent.qlearn.save(f"Q_Learner_{max_episodes}.csv")

if __name__ == "__main__":
    main()