import random
import pandas as pd
import numpy as np
from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import features, actions
import sys
import absl.flags as flags
import os

# Constants for actions
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Constants for features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Constants for identification
_PLAYER_SELF = 1
_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

# Constants for actions queue
_NOT_QUEUED = [0]
_QUEUED = [1]

# Action definitions
ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

# List of all actions for the agent
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

# Reward values
KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list of actions
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
            self.q_table = pd.concat([self.q_table, pd.DataFrame(new_row).T])

    def save(self, filename):
        """Save the Q-table to a CSV file, ensuring the directory exists."""
        full_path = os.path.join('Building a Smart Agent', 'Q-tables', filename)
        directory = os.path.dirname(full_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        self.q_table.to_csv(full_path)
    
class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartAgent, self).__init__()
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_action = None
        self.previous_state = None

    def step(self, obs):
        super(SmartAgent, self).step(obs)
        
        player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
        unit_type = obs.observation['feature_screen'][_UNIT_TYPE]

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        supply_depot_count = 1 if depot_y.any() else 0
        barracks_count = 1 if barracks_y.any() else 0

        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]

        current_state = [
            supply_depot_count,
            barracks_count,
            supply_limit,
            army_supply
        ]

        if self.previous_action is not None:
            reward = 0

            killed_unit_score = obs.observation['score_cumulative'][5]
            killed_building_score = obs.observation['score_cumulative'][6]

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_killed_unit_score = obs.observation['score_cumulative'][5]
        self.previous_killed_building_score = obs.observation['score_cumulative'][6]
        self.previous_state = current_state
        self.previous_action = rl_action

        return self.perform_action(obs, smart_action)

    def perform_action(self, obs, smart_action):
        if smart_action == ACTION_DO_NOTHING:
            return actions.FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_SCV:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]

                return actions.FUNCTIONS.select_point("select", target)

        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                if unit_y.any():
                    target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)

                    return actions.FUNCTIONS.Build_SupplyDepot_screen("now", target)

        elif smart_action == ACTION_BUILD_BARRACKS:
            if _BUILD_BARRACKS in obs.observation['available_actions']:
                unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

                if unit_y.any():
                    target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)

                    return actions.FUNCTIONS.Build_Barracks_screen("now", target)

        elif smart_action == ACTION_SELECT_BARRACKS:
            unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()

            if unit_y.any():
                target = [int(unit_x.mean()), int(unit_y.mean())]

                return actions.FUNCTIONS.select_point("select", target)

        elif smart_action == ACTION_BUILD_MARINE:
            if _TRAIN_MARINE in obs.observation['available_actions']:
                return actions.FUNCTIONS.Train_Marine_quick("queued")

        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FUNCTIONS.select_army("select")

        elif smart_action == ACTION_ATTACK:
            if _ATTACK_MINIMAP in obs.observation['available_actions']:
                if self.base_top_left:
                    return actions.FUNCTIONS.Attack_minimap("now", [39, 45])

                return actions.FUNCTIONS.Attack_minimap("now", [21, 24])

        return actions.FUNCTIONS.no_op()

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]
    
    # def __del__(self):
    #     """Ensure the Q-table is saved when the agent is deleted or program ends."""
    #     self.qlearn.save()

def main():
    max_episodes = 1
    # Parse flags
    flags.FLAGS(sys.argv)

    agent = SmartAgent()
    try:
        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True
            ),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=True
        ) as env:
            
            run_loop.run_loop([agent], env, max_episodes=1)

    except KeyboardInterrupt:
        pass
    finally:
        agent.qlearn.save(f"Q_table_maxep_{max_episodes}.csv")


if __name__ == "__main__":
    main()