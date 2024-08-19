import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import features, actions
import sys
import absl.flags as flags
import os
import argparse

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

# DQN model definition
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

# Replay Memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Base Agent class for common functionalities
class Agent(base_agent.BaseAgent):
    def __init__(self):
        super(Agent, self).__init__()
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_action = None
        self.previous_state = None
        self.base_top_left = None

    def transform_location(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        return [x + x_distance, y + y_distance]

    def compute_reward(self, obs):
        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        reward = 0

        if self.previous_action is not None:
            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD
            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        return reward

    def get_current_state(self, obs):
        player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        supply_depot_count = 1 if depot_y.any() else 0
        barracks_count = 1 if barracks_y.any() else 0
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        return [supply_depot_count, barracks_count, supply_limit, army_supply]

    def perform_action(self, obs, smart_action):        
        if smart_action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])

        elif smart_action == ACTION_SELECT_SCV:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                
            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        
        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                
                if unit_y.any():
                    target = self.transform_location(int(unit_x.mean()), 0, int(unit_y.mean()), 20)
                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
        
        elif smart_action == ACTION_BUILD_BARRACKS:
            if _BUILD_BARRACKS in obs.observation['available_actions']:
                unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                
                if unit_y.any():
                    target = self.transform_location(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
                    return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

        elif smart_action == ACTION_SELECT_BARRACKS:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
                
            if unit_y.any():
                target = [int(unit_x.mean()), int(unit_y.mean())]
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        
        elif smart_action == ACTION_BUILD_MARINE:
            if _TRAIN_MARINE in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        
        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        
        elif smart_action == ACTION_ATTACK:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])
        
        return actions.FunctionCall(_NO_OP, [])

# SmartAgent class for DQN implementation
class SmartAgent(Agent):
    def __init__(self, load_model=False, model_path=None):
        super(SmartAgent, self).__init__()
        
        # Hyperparameters
        self.gamma = 0.9
        self.lr = 0.001
        self.epsilon = 1.0 if not load_model else 0.0  # Exploration during training, no exploration during testing
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.sync_rate = 10
        
        # DQN
        input_dim = 4  # State: [supply_depot_count, barracks_count, supply_limit, army_supply]
        hidden_dim = 128
        output_dim = len(smart_actions)
        
        self.policy_dqn = DQN(input_dim, hidden_dim, output_dim)
        self.target_dqn = DQN(input_dim, hidden_dim, output_dim)

        if load_model and model_path:
            self.load(model_path)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        
        # Replay Memory
        self.memory = ReplayMemory(maxlen=1000)
        self.sync_counter = 0

    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.choice(range(len(smart_actions)))
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_dqn(state_tensor)
            return torch.argmax(q_values).item()

    def replay_experience(self):
        # Sample mini-batch from replay memory
        mini_batch = self.memory.sample(self.batch_size)

        # Prepare batch data
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)

        # Compute current Q values
        current_q_values = self.policy_dqn(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            max_next_q_values = self.target_dqn(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values)

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync target network
        self.sync_counter += 1
        if self.sync_counter % self.sync_rate == 0:
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.policy_dqn.state_dict(), filename)
        print(f"Model successfully saved to {filename}")

    def load(self, filename):
        self.policy_dqn.load_state_dict(torch.load(filename))
        self.policy_dqn.eval()  # Set the network to evaluation mode
        print(f"Model successfully loaded from {filename}")

# Main function for both training and testing
def main(mode="train"):
    # Parse absl flags separately
    flags.FLAGS(sys.argv[:1])  # Only pass the program name to absl.flags
    max_episodes = 100 if mode == "train" else 10
    agent = SmartAgent(load_model=(mode == "test"), model_path="dqn_agent.pth" if mode == "test" else None)

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
            run_loop.run_loop([agent], env, max_episodes=max_episodes)
    except KeyboardInterrupt:
        print(f"Game interrupted in {mode} mode.")
    finally:
        if mode == "train":
            agent.save("dqn_agent.pth")
        print(f"{mode.capitalize()} mode has ended.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Choose whether to train or test the agent.")
    
    # Parse known arguments with argparse
    args, unknown = parser.parse_known_args()

    # Remove the --mode flag from sys.argv so absl.flags doesn't see it
    sys.argv = sys.argv[:1] + unknown

    # Run the main function with the selected mode
    main(mode=args.mode)
