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

max_episodes = 1000  # Define the maximum episodes variable

# Constants for setting up the logging directory and filename
log_dir = 'TBQL_logs'
log_file_name = f'Q_Learner_logs_{max_episodes}.log'
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

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        """
        Initialize the Q-learning table.
        
        Parameters:
            actions (list): List of possible actions.
            learning_rate (float): Learning rate (alpha).
            reward_decay (float): Discount factor (gamma).
            e_greedy (float): Exploration-exploitation tradeoff parameter (epsilon).
        """
        self.actions = actions  # a list of actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.disallowed_actions = {}

    def choose_action(self, observation, excluded_actions=[]):
        """
        Choose an action based on the state (observation).

        Parameters:
            observation: The current state.
            excluded_actions (list): Actions that should be excluded in the current state.

        Returns:
            action: Selected action as a response to the current state.
        """
        self.check_state_exist(observation)
        state_action = self.q_table.loc[observation, :]
        
        # Exclude the disallowed actions
        state_action = state_action.drop(labels=excluded_actions, errors='ignore')

        # Handling the case where no actions are available
        if state_action.empty:
            logging.warning(f"No available actions for state {observation}, performing NO_OP")
            return 'donothing'  # Assume there is an action defined as 'donothing'

        if np.random.uniform() < self.epsilon:
            # Choosing the best action based on Q-values
            action = state_action.idxmax()
        else:
            # Choosing a random action
            action = np.random.choice(state_action.index)

        logging.info(f"Choosing action {action} for state {observation}")
        return action

    def learn(self, s, a, r, s_):
        """
        Update the Q-table using the learning rule based on the transition.

        Parameters:
            s: Current state.
            a: Action taken.
            r: Reward received.
            s_: Next state.
        """
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # Bellman equation
        else:
            q_target = r  # No future reward if next state is terminal
        
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # Update Q-value
        logging.info(f"Updated Q-value for state {s}, action {a}: {q_predict} -> {q_target}")

    def check_state_exist(self, state):
        """
        Check if a state exists in the Q-table, and if not, add it.

        Parameters:
            state: The state to check in the Q-table.
        """
        if state not in self.q_table.index:
            # Append new state to q table
            new_row = pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = pd.concat([self.q_table, pd.DataFrame(new_row).T])

    def save(self, filename):
        """
        Save the Q-table to a CSV file.

        Parameters:
            filename: The name of the file to save the Q-table to.
        """
        full_path = os.path.join('TBQL_Q-tables', filename)
        directory = os.path.dirname(full_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        self.q_table.to_csv(full_path)
        
class Agent(base_agent.BaseAgent):
    def __init__(self):
        super(Agent, self).__init__()        
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))        
        self.previous_action = None
        self.previous_state = None        
        self.cc_y = None
        self.cc_x = None
        self.move_number = 0
        logging.info("Initializing the SparseAgent")
        
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
        logging.info("Processing a new step")
        super(Agent, self).step(obs)
        
        if obs.last():
            reward = obs.reward
        
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
            
            self.previous_action = None
            self.previous_state = None
            
            self.move_number = 0
            
            return actions.FunctionCall(_NO_OP, [])
        
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

        if obs.first():
            player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
            self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0
        
        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))
            
        supply_used = obs.observation['player'][3]
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        worker_supply = obs.observation['player'][6]
        
        supply_free = supply_limit - supply_used
        
        if self.move_number == 0:
            self.move_number += 1
            
            current_state = np.zeros(12)
            current_state[0] = cc_count
            current_state[1] = supply_depot_count
            current_state[2] = barracks_count
            current_state[3] = obs.observation['player'][_ARMY_SUPPLY]
    
            hot_squares = np.zeros(4)        
            enemy_y, enemy_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 32))
                x = int(math.ceil((enemy_x[i] + 1) / 32))
                
                hot_squares[((y - 1) * 2) + (x - 1)] = 1
            
            if not self.base_top_left:
                hot_squares = hot_squares[::-1]
            
            for i in range(0, 4):
                current_state[i + 4] = hot_squares[i]
    
            green_squares = np.zeros(4)        
            friendly_y, friendly_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            for i in range(0, len(friendly_y)):
                y = int(math.ceil((friendly_y[i] + 1) / 32))
                x = int(math.ceil((friendly_x[i] + 1) / 32))
                
                green_squares[((y - 1) * 2) + (x - 1)] = 1
            
            if not self.base_top_left:
                green_squares = green_squares[::-1]
            
            for i in range(0, 4):
                current_state[i + 8] = green_squares[i]
    
            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))
        
            excluded_actions = []
            if supply_depot_count == 2 or worker_supply == 0:
                excluded_actions.append(1)
                
            if supply_depot_count == 0 or barracks_count == 2 or worker_supply == 0:
                excluded_actions.append(2)

            if supply_free == 0 or barracks_count == 0:
                excluded_actions.append(3)
                
            if army_supply == 0:
                excluded_actions.append(4)
                excluded_actions.append(5)
                excluded_actions.append(6)
                excluded_actions.append(7)
        
            rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)

            self.previous_state = current_state
            self.previous_action = rl_action
        
            smart_action, x, y = self.splitAction(self.previous_action)
            
            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                    
                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]
                    
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
                
            elif smart_action == ACTION_BUILD_MARINE:
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]
            
                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
                
            elif smart_action == ACTION_ATTACK:
                if _SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        
        elif self.move_number == 1:
            self.move_number += 1
            
            smart_action, x, y = self.splitAction(self.previous_action)
                
            if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if supply_depot_count < 2 and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                    if self.cc_y.any():
                        if supply_depot_count == 0:
                            target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
                        elif supply_depot_count == 1:
                            target = self.transformDistance(round(self.cc_x.mean()), -25, round(self.cc_y.mean()), -25)
    
                        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
            
            elif smart_action == ACTION_BUILD_BARRACKS:
                if barracks_count < 2 and _BUILD_BARRACKS in obs.observation['available_actions']:
                    if self.cc_y.any():
                        if  barracks_count == 0:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
                        elif  barracks_count == 1:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)
    
                        return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
    
            elif smart_action == ACTION_BUILD_MARINE:
                if _TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        
            elif smart_action == ACTION_ATTACK:
                do_it = True
                
                if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
                    do_it = False
                
                if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
                    do_it = False
                
                if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)
                    
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(int(x) + (x_offset * 8), int(y) + (y_offset * 8))])
                
        elif self.move_number == 2:
            self.move_number = 0
            
            smart_action, x, y = self.splitAction(self.previous_action)
                
            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if _HARVEST_GATHER in obs.observation['available_actions']:
                    unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()
                    
                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)
                        
                        m_x = unit_x[i]
                        m_y = unit_y[i]
                        
                        target = [int(m_x), int(m_y)]
                        
                        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])
        
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