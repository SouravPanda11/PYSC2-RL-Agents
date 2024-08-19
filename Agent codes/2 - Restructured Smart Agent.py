import random
import numpy as np
import pandas as pd
from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import features, actions
import sys
import absl.flags as flags
import os

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

# Q-Learning Table
class QLearningTable:
    
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)
        q_predict = self.q_table.loc[s, a]
        q_target = r + self.gamma * self.q_table.loc[s_, :].max() 
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            new_row = pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = pd.concat([self.q_table, new_row.to_frame().T])
    
    def save(self, filename):
        directory = 'Q-tables'
        full_path = os.path.join(directory, filename)
        directory = os.path.dirname(full_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        self.q_table.to_csv(full_path)
        print(f"Q-table successfully saved to {full_path}")

class Agent(base_agent.BaseAgent):
    def __init__(self):
        super(Agent, self).__init__()
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_action = None
        self.previous_state = None

    def transformLocation(self, x, x_distance, y, y_distance):
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

        # Select an SCV unit from the screen.
        elif smart_action == ACTION_SELECT_SCV:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                
            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        
        # Build a Supply Depot at an appropriate location near the Command Center.
        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                
                if unit_y.any():
                    target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)
                
                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
        
        # Build Barracks near the Command Center.
        elif smart_action == ACTION_BUILD_BARRACKS:
            if _BUILD_BARRACKS in obs.observation['available_actions']:
                unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                
                if unit_y.any():
                    target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
            
                    return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

        # Select existing Barracks to produce Marines.
        elif smart_action == ACTION_SELECT_BARRACKS:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
                
            if unit_y.any():
                target = [int(unit_x.mean()), int(unit_y.mean())]
        
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        
        # Order the selected Barracks to train a Marine.
        elif smart_action == ACTION_BUILD_MARINE:
            if _TRAIN_MARINE in obs.observation['available_actions']:
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        
        # Select all army units.
        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        
        # Command the selected military units to attack a specific point on the minimap.
        elif smart_action == ACTION_ATTACK:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])
            
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])
        
        return actions.FunctionCall(_NO_OP, [])

class SmartAgent(Agent):
    def __init__(self):
        super(SmartAgent, self).__init__()
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

    def step(self, obs):
        super(SmartAgent, self).step(obs)
        current_state = self.get_current_state(obs)
        reward = self.compute_reward(obs)

        if self.previous_action is not None:
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_state = current_state
        self.previous_action = rl_action
        return self.perform_action(obs, smart_action)

def main():
    max_episodes = 2
    flags.FLAGS(sys.argv)
    agent = SmartAgent()
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
        print("Game interrupted by user.")
    finally:
        agent.qlearn.save(f"Smart_Agent_Q_table_maxep_{max_episodes}.csv")
        print("Game has ended.")

if __name__ == "__main__":
    main()