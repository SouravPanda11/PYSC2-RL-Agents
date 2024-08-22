import random
import pandas as pd
import numpy as np
from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import features, actions
import sys
import absl.flags as flags
import os
import math

# Constants for actions
# Identifier for the action that represents 'no operation', used when no other action is taken in a game step.
_NO_OP = actions.FUNCTIONS.no_op.id
# Identifier for the action to select a unit or structure at a specified point on the screen.
_SELECT_POINT = actions.FUNCTIONS.select_point.id
# Identifier for the action to build a Supply Depot on the game screen, a structure that increases the supply limit.
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
# Identifier for the action to build a Barracks on the game screen, an essential military production building.
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
# Identifier for the quick action to train a Marine, a basic infantry unit, from a Barracks.
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
# Identifier for the action to select all combat units in the player's current view.
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
# Identifier for the action to order selected military units to attack a specific point on the minimap.
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
# Index for the 'player_relative' feature layer in screen features, identifying player ownership of units/structures.
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
# Index for the 'unit_type' feature layer in screen features, indicating the type of each unit or structure visible on screen.
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Constants for identification
# Identifier for the player's own units and structures within the game environment.
_PLAYER_SELF = 1
# Identifier for enemy units and structures on the minimap, used to detect and interact with hostile elements.
_PLAYER_HOSTILE = 4

# Unit IDs
# Unit identifier for Terran Barracks, used for training infantry units like Marines.
_TERRAN_BARRACKS = 21
# Unit identifier for Terran Command Center, the primary structure for base operations and worker unit production.
_TERRAN_COMMANDCENTER = 18
# Unit identifier for Terran Supply Depot, essential for increasing the unit supply limit to build more units.
_TERRAN_SUPPLY_DEPOT = 19
# Unit identifier for Terran SCV, the worker unit responsible for resource gathering and building construction.
_TERRAN_SCV = 45

# Constants for actions queue
# Command modifier indicating that an action should not be queued but executed immediately.
_NOT_QUEUED = [0]
# Command modifier indicating that an action should be added to the current queue of actions.
_QUEUED = [1]

# Action definitions
# Definition of action constants
# Represents an action where the agent performs no operation during a game tick.
ACTION_DO_NOTHING = 'donothing' 
# Specifies the action to select an SCV (Space Construction Vehicle), which is the basic worker unit for the Terran race in StarCraft II.
ACTION_SELECT_SCV = 'selectscv' 
# Defines the action to build a Supply Depot, a Terran structure that increases the supply limit, allowing for the production of additional units.
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
# Sets the action to construct Barracks, which are primary military buildings for producing infantry units like Marines.
ACTION_BUILD_BARRACKS = 'buildbarracks'
# Represents the action to select an existing Barracks. Selecting buildings is typically a precursor to issuing production commands.
ACTION_SELECT_BARRACKS = 'selectbarracks'
# Commands the agent to train a Marine from the selected Barracks. Marines are versatile infantry units essential to early-game defense and aggression.
ACTION_BUILD_MARINE = 'buildmarine'
# This action commands the agent to select all military units under its control, facilitating grouped movements and coordinated attacks.
ACTION_SELECT_ARMY = 'selectarmy'
# Instructs the selected units to perform an attack move, typically directed towards a location on the minimap, enabling offensive strategies.
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
]

# Generate attack actions for a reduced set of critical points on the mini-map.
# The map is divided into a 4x4 grid, and attack actions are created for the center of each 16x16 block.
# This approach reduces the complexity of the action space, focusing on strategically relevant areas of the map.
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        # Ensure that the actions are created for every 16th coordinate on both x and y axes,
        # effectively creating a grid. The attack coordinates are then centered by subtracting 8,
        # positioning the action in the middle of each 16x16 block.
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

# Reward values for specific in-game achievements
# Assigns a reward value for each enemy unit killed by the agent. This incentivizes the agent to engage in combat and eliminate enemy forces.
KILL_UNIT_REWARD = 0.2
# Sets a higher reward value for destroying enemy buildings, which are typically more valuable targets than individual units.
KILL_BUILDING_REWARD = 0.5

# Q-Learning Table
class QLearningTable:
    """
    Implements a Q-learning algorithm with a table of Q-values for each (state, action) pair.
    """
    
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        """
        Initializes the Q-Learning table for the agent.

        Parameters:
        actions (list): List of all possible actions the agent can take.
        learning_rate (float): The rate at which the agent should learn from new experiences.
        reward_decay (float): The discount factor (gamma), which quantifies how much importance is given to future rewards.
        e_greedy (float): The probability of choosing a random action instead of the best action.
        """
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


    def choose_action(self, observation):
        """
        Selects an action to perform based on the current observation (state) using an epsilon-greedy policy.
        
        Parameters:
        observation (any): The current observed state.

        Returns:
        action (any): The action chosen to perform next.
        """
        
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            # Exploit by selecting the best-known action based on Q-values
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):
        """
        Updates the Q-value table using the latest experience tuple.

        Parameters:
        s (any): The initial state.
        a (any): The action taken.
        r (float): The reward received after taking action 'a' from state 's'.
        s_ (any): The new state reached after taking action 'a'.
        """
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.loc[s, a]
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # Calculate the maximum Q-value for the next state
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # Update Q-value using the learning rate

    def check_state_exist(self, state):
        """
        Ensures the state exists in the Q-table. If not, adds a new row with initialized values.

        Parameters:
        state (any): The state to check or add to the Q-table.
        """
        if state not in self.q_table.index:
            # Create a new DataFrame for the missing state with initialized Q-values
            new_row = pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state)
            # Append new state to q table using concat instead of deprecated append
            self.q_table = pd.concat([self.q_table, new_row.to_frame().T])
    
    def save(self, filename):
        """
        Save the Q-table to a CSV file, ensuring the directory exists.

        Parameters:
        filename (str): The name of the file where the Q-table should be saved.
        """
        # Define the path where the Q-tables directory will be located.
        directory = 'Q-tables'
        full_path = os.path.join(directory, filename)
        
        # Extract the directory path and check if it exists, create if it doesn't.
        directory = os.path.dirname(full_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)  # Safe way to create the directory.
        
        # Save the DataFrame to CSV format at the specified location.
        self.q_table.to_csv(full_path)

        print(f"Q-table successfully saved to {full_path}")
    
class AttackAgent(base_agent.BaseAgent):
    def __init__(self):
        super(AttackAgent, self).__init__()
        # Initialize the Q-learning table with the range of action indices.
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        # Track scores for killed units and buildings to calculate rewards.
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        # Initialize previous action and state for learning from past experiences.
        self.previous_action = None
        self.previous_state = None
        
    def transformDistance(self, x, x_distance, y, y_distance):
        """
        Adjusts the position based on the relative distance from a reference point,
        considering whether the base is located at the top left or not.
        
        Args:
        x (int): Base x-coordinate on the screen or minimap.
        x_distance (int): Horizontal distance to offset from the base x-coordinate.
        y (int): Base y-coordinate on the screen or minimap.
        y_distance (int): Vertical distance to offset from the base y-coordinate.
        
        Returns:
        list: A list containing the transformed [x, y] coordinates.
        """
        # If the base is not at the top left, adjust by subtracting the distances.
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        # If the base is at the top left, adjust by adding the distances.
        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        """
        Converts absolute map coordinates into relative coordinates based on the base's location.
        
        Args:
        x (int): The absolute x-coordinate on the minimap.
        y (int): The absolute y-coordinate on the minimap.
        
        Returns:
        list: A list containing the transformed [x, y] coordinates relative to the base location.
        """
        # If the base is not at the top left, mirror the coordinates to the opposite side.
        if not self.base_top_left:
            return [64 - x, 64 - y]
        # If the base is at the top left, use the coordinates as they are.
        return [x, y]

        
    def step(self, obs):
        """
        Process an observation from the game, decide on an action, and update the agent's state.
        
        Args:
        obs: A data structure that contains observations from the game environment.
        
        Returns:
        An action to be executed in the game environment.
        """
        # Inherit functionalities from the parent class
        super(AttackAgent, self).step(obs)
        
        # Determine the player's base location to adjust strategies based on map position.
        player_y, player_x = (obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
        # Access unit type data from the screen to detect various structures and units.
        screen_data = obs.observation['feature_screen']
        unit_type = screen_data[_UNIT_TYPE]
        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        supply_depot_count = 1 if depot_y.any() else 0
        barracks_count = 1 if barracks_y.any() else 0
        
        # Retrieve additional player-related information from the observation.
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        
        # Construct a vector that represents the current state with information gathered.
        current_state = np.zeros(20)
        current_state[0] = supply_depot_count
        current_state[1] = barracks_count
        current_state[2] = supply_limit
        current_state[3] = army_supply
        
        # Create a grid to mark enemy presence on the minimap for better attack decisions.        
        hot_squares = np.zeros(16)
        enemy_y, enemy_x = (screen_data[_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))

            # Adjust for zero-based index array and ensure index does not go out of bounds
            index = (y - 1) * 4 + (x - 1)
            if index >= 0 and index < 16:
                hot_squares[index] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]
            
        # Learn from the previous action based on the rewards collected and update the Q-table.
        if self.previous_action is not None:
            reward = 0
            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD
            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD
                
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))
            print(f"Learned: {self.previous_action} -> Reward: {reward}")
            
        # Decide the next action to take based on the current state and Q-learning table.
        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]
        print(f"Action chosen: {smart_action}")
        
        # Update internal state tracking variables for next learning step.
        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_state = current_state
        self.previous_action = rl_action
        
        # Execute the chosen action.
        return self.perform_action(obs, smart_action)
            
    def perform_action(self, obs, smart_action):
        """
        Execute the chosen action within the game environment.

        Args:
        obs: The current game state observations.
        smart_action: The action selected by the agent to perform.

        Returns:
        A StarCraft II action function call to be executed in the game.
        """
        x = 0
        y = 0
        # Parse the coordinates from the action if it's an attack action.
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')
        
        # Perform a no-operation, essentially skipping the turn.    
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
                    target = self.transformDistance(int(unit_x.mean()), 0, int(unit_y.mean()), 20)
                
                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
        
        # Build Barracks near the Command Center.
        elif smart_action == ACTION_BUILD_BARRACKS:
            if _BUILD_BARRACKS in obs.observation['available_actions']:
                unit_type = obs.observation['feature_screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                
                if unit_y.any():
                    target = self.transformDistance(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
            
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
            if (obs.observation['single_select'].any() or obs.observation['multi_select'].any()) and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [int(x), int(y)]])
            else:
                print("No unit selected for attack or action not available")

        return actions.FunctionCall(_NO_OP, [])

def main():
    """
    Main function to set up and run a StarCraft II environment with the AttackAgent.
    This function configures the game environment, initializes the agent, and manages the game loop.
    """
    max_episodes=1 # Number of episodes to run the game for.
     
    # Parse command line arguments that the script was called with; needed for proper SC2Env configuration.
    flags.FLAGS(sys.argv)

    # Instantiate the SimpleAgent.
    agent = AttackAgent()

    try:
        # Setup the StarCraft II environment for the agent using sc2_env.SC2Env.
        # The environment specifies how the game will be set up and run.
        with sc2_env.SC2Env(
            map_name="Simple64",  # Name of the map on which the game will be played.
            players=[
                sc2_env.Agent(sc2_env.Race.terran),  # Define the agent's race as Terran.
                sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)  # Add a very easy bot opponent of random race.
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),  # Set the resolution for screen and minimap features.
                use_feature_units=True  # Allow the agent to access detailed feature units directly.
            ),
            step_mul=8,  # Number of game steps to take before the agent is called again.
            game_steps_per_episode=0,  # Unlimited game steps per episode until the end condition is met.
            visualize=True  # Enable visualization to see the game while the agent plays.
        ) as env:
            # Run the agent within the environment for one episode.
            # The run_loop handles the interaction between the agent and the environment.
            run_loop.run_loop([agent], env, max_episodes=max_episodes)
    except KeyboardInterrupt:
        # If the user interrupts the game (e.g., with Ctrl+C), print a message.
        print("Game interrupted by user.")
    finally:
        # Save the Q-table to a CSV file when the game ends.
        agent.qlearn.save(f"Attack_Agent_Q_table_maxep_{max_episodes}.csv")
        # Print a message indicating the game has ended.
        print("Game has ended.")

# This line checks if this script is being run as the main program and not imported as a module.
if __name__ == "__main__":
    main()