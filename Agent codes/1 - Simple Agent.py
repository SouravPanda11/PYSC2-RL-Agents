from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import features, actions
import sys
import absl.flags as flags

import time # Importing the time library to use sleep functionality

# Functions
# Identifier for the action to build a Barracks on the game screen, an essential military production building.
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
# Identifier for the action to build a Supply Depot on the game screen, a structure that increases the supply limit.
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
# Identifier for the action that represents 'no operation', used when no other action is taken in a game step.
_NOOP = actions.FUNCTIONS.no_op.id
# Identifier for the action to select a unit or structure at a specified point on the screen.
_SELECT_POINT = actions.FUNCTIONS.select_point.id
# Identifier for the quick action to train a Marine, a basic infantry unit, from a Barracks.
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
# Identifier for the action to set a rally point for units on the minimap, directing where units go after production.
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
# Identifier for the action to select all combat units in the player's current view.
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
# Identifier for the action to order selected military units to attack a specific point on the minimap.
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
# Index for the 'player_relative' feature layer in screen features, identifying player ownership of units/structures.
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
# Index for the 'unit_type' feature layer in screen features, indicating the type of each unit or structure visible on screen.
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
# Unit identifier for Terran Barracks, used for training infantry units like Marines.
_TERRAN_BARRACKS = 21
# Unit identifier for Terran Command Center, the primary structure for base operations and worker unit production.
_TERRAN_COMMANDCENTER = 18
# Unit identifier for Terran SCV, the worker unit responsible for resource gathering and building construction.
_TERRAN_SCV = 45

# Parameters
# Identifier for the player's own units and structures within the game environment.
_PLAYER_SELF = 1
# Game parameter for tracking the amount of supply currently in use by the player.
_SUPPLY_USED = 3
# Game parameter for tracking the maximum supply limit the player can utilize.
_SUPPLY_MAX = 4
# Command modifier indicating that an action should not be queued but executed immediately.
_NOT_QUEUED = [0]
# Command modifier indicating that an action should be added to the current queue of actions.
_QUEUED = [1]

class SimpleAgent(base_agent.BaseAgent):
    """
    An AI agent designed to play StarCraft II within the PySC2 framework. This agent is capable of basic game tasks such as
    building essential structures (Supply Depot and Barracks), managing SCVs, and handling military operations including setting rally points
    and commanding an army. It maintains internal state flags to manage and coordinate these activities throughout the game.
    
    Attributes:
        base_top_left (bool): Tracks whether the base is positioned at the top-left of the map.
        supply_depot_built (bool): Indicates whether a Supply Depot has been constructed.
        scv_selected (bool): Indicates whether an SCV has been selected.
        barracks_built (bool): Indicates whether a Barracks has been constructed.
        barracks_selected (bool): Indicates whether a Barracks has been selected.
        barracks_rallied (bool): Indicates whether a rally point for the Barracks has been set.
        army_selected (bool): Indicates whether the army has been selected.
        army_rallied (bool): Indicates whether the army has been rallied to a specific point.
    """
    
    # Initialization of variables to track the state of game-specific tasks
    # Variable to store the starting base location; 'None' indicates it has not yet been determined.
    base_top_left = None
    # Flag to indicate whether a Supply Depot has been built; initialized to False.
    supply_depot_built = False
    # Flag to indicate whether an SCV (worker unit) has been selected; initialized to False.
    scv_selected = False
    # Flag to indicate whether a Barracks has been built; initialized to False.
    barracks_built = False
    # Flag to indicate whether a Barracks has been selected; initialized to False.
    barracks_selected = False
    # Flag to indicate whether a rally point for the Barracks has been set; initialized to False.
    barracks_rallied = False
    # Flag to indicate whether the army has been selected; initialized to False.
    army_selected = False
    # Flag to indicate whether the army has been rallied to a specific point; initialized to False.
    army_rallied = False


    def transformLocation(self, x, x_distance, y, y_distance):
        """
        Transforms a given location based on the base's relative position on the map.
        
        This method adjusts the target location by applying a specified distance offset,
        allowing for accurate placement of buildings or movement of units relative to the base's location.
        If the base is in the top left (base_top_left is True), the method adds the distances to the coordinates;
        if the base is not in the top left (base_top_left is False), it subtracts the distances,
        effectively mirroring the placement to the opposite side of the map.

        Parameters:
        x (int): The base x-coordinate on the screen or minimap.
        x_distance (int): The horizontal distance to offset from the base x-coordinate.
        y (int): The base y-coordinate on the screen or minimap.
        y_distance (int): The vertical distance to offset from the base y-coordinate.

        Returns:
        list: A list containing the transformed [x, y] coordinates.
        """
        if not self.base_top_left:
            # If base is at the bottom right, subtract distances to mirror the action to the opposite side.
            return [x - x_distance, y - y_distance]
        
        # If base is at the top left, add distances to move the action outward from the base point.
        return [x + x_distance, y + y_distance]

    
    def step(self, obs):
        """
        Executes agent actions at each step of the game based on the current observation of the game state.
        This method is called continuously during the game to update and execute the agent's strategy.

        Parameters:
        obs (dict): The current game state observations from the StarCraft II environment.
        """
        # Call the base class's step method to perform any setup or common functions.
        super(SimpleAgent, self).step(obs)

        # Introduce a brief pause to simulate more human-like action intervals.
        time.sleep(0.5)

        # Determine the initial base position if it hasn't been set yet.
        if self.base_top_left is None:
            # Extract player's position data from the minimap using the player_relative feature.
            player_y, player_x = (obs.observation["feature_minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            # Calculate the mean of the y-coordinates to determine if the base is in the top left side of the map.
            self.base_top_left = player_y.mean() <= 31

        # Logic to build a supply depot if it has not been built yet.
        if not self.supply_depot_built:
            # Check if an SCV is selected to build the supply depot.
            if not self.scv_selected:
                # Get the SCV unit type positions from the screen observation.
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                # Select the first SCV found.
                target = [unit_x[0], unit_y[0]]
                self.scv_selected = True
                # Command to select the SCV on the screen.
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:
                # Find the command center to place the supply depot nearby.
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                # Use the transformLocation method to position the supply depot appropriately.
                target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)
                self.supply_depot_built = True
                # Command to build the supply depot.
                return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])

        # Logic to build barracks if they have not been built yet.
        elif not self.barracks_built:
            if _BUILD_BARRACKS in obs.observation["available_actions"]:
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
                self.barracks_built = True
                # Command to build the barracks.
                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

        # Logic to set a rally point for the barracks.
        elif not self.barracks_rallied:
            if not self.barracks_selected:
                unit_type = obs.observation["feature_screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]
                    self.barracks_selected = True
                    # Command to select the barracks.
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            else:
                self.barracks_rallied = True
                # Set the rally point depending on the base location.
                if self.base_top_left:
                    return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 21]])
                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 46]])

        # Logic to train marines if there is enough supply.
        elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and _TRAIN_MARINE in obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

        # Logic to control and command the army.
        elif not self.army_rallied:
            if not self.army_selected:
                if _SELECT_ARMY in obs.observation["available_actions"]:
                    self.army_selected = True
                    self.barracks_selected = False
                    # Command to select all combat units.
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
                self.army_rallied = True
                self.army_selected = False
                # Command the army to attack a specific point depending on the base location.
                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])

        # Default action if no other conditions are met.
        return actions.FunctionCall(_NOOP, [])

def main():
    """
    Main function to set up and run a StarCraft II environment with the SimpleAgent.
    This function configures the game environment, initializes the agent, and manages the game loop.
    """
    # Parse command line arguments that the script was called with; needed for proper SC2Env configuration.
    flags.FLAGS(sys.argv)

    # Instantiate the SimpleAgent.
    agent = SimpleAgent()

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
            run_loop.run_loop([agent], env, max_episodes=1)
    except KeyboardInterrupt:
        # If the user interrupts the game (e.g., with Ctrl+C), print a message.
        print("Game interrupted by user.")
    finally:
        # This block ensures that the environment is properly cleaned up after the game finishes or is interrupted.
        print("Game has ended.")

# This line checks if this script is being run as the main program and not imported as a module.
if __name__ == "__main__":
    main()


