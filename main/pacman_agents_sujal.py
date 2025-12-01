# pacman_agents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import game
import util


class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def get_action(self, state):
        legal = state.get_legal_pacman_actions()
        current = state.get_pacman_state().configuration.direction
        if current == Directions.STOP:
            current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal:
            return left
        if current in legal:
            return current
        if Directions.RIGHT[current] in legal:
            return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal:
            return Directions.LEFT[left]
        return Directions.STOP


class GreedyAgent(Agent):
    def __init__(self, eval_fn="score_evaluation"):
        if isinstance(eval_fn, str):
            self.evaluation_function = util.lookup(eval_fn, globals())
        else:
            self.evaluation_function = eval_fn
        assert self.evaluation_function is not None
        # --- NEW: Add memory for the last action ---
        self.last_action = None
        # --- NEW: Add memory for visited positions to prevent oscillation ---
        self.visited_history = [] 

    def get_action(self, state):
        # Generate candidate actions
        legal = state.get_legal_pacman_actions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        successors = [(state.generate_successor(0, action), action) for action in legal]
        
        # Evaluate successors
        scored = []
        for next_state, action in successors:
            score = self.evaluation_function(next_state)
            
            # --- OSCILLATION PREVENTION ---
            # Penalize revisiting recent positions
            pacman_pos = next_state.get_pacman_position()
            if pacman_pos in self.visited_history:
                # Apply a penalty based on how recently it was visited
                # More recent = higher penalty
                recency = self.visited_history.index(pacman_pos)
                penalty = (len(self.visited_history) - recency) * 10 
                score -= penalty
            
            scored.append((score, action))

        # Find the best score and all actions that lead to it
        best_score = max(scored)[0]
        best_actions = [pair[1] for pair in scored if pair[0] == best_score]

        # --- THIS IS THE FIX FOR OSCILLATION ---
        # If there's more than one "best" action, and we have a last action,
        # try to remove the action that would reverse our last move.
        if len(best_actions) > 1 and self.last_action is not None:
            reverse_action = Directions.REVERSE[self.last_action]
            if reverse_action in best_actions:
                best_actions.remove(reverse_action)
        # --- END OF FIX ---

        # Choose one of the remaining best actions randomly
        chosen_action = random.choice(best_actions)
        
        # Remember the chosen action for the next turn
        self.last_action = chosen_action
        
        # Update visited history
        current_pos = state.get_pacman_position()
        self.visited_history.append(current_pos)
        if len(self.visited_history) > 10: # Keep last 10 positions
            self.visited_history.pop(0)
        
        return chosen_action

def score_evaluation(state):
    return state.get_score()


import time
import genetic_algorithm  # Import the new GA file
from util import manhattan_distance


def genetic_evaluation_function(current_game_state, weights):
    """
    --- FINAL, SIMPLIFIED EVALUATION FUNCTION ---
    Focuses on the two most important goals: SURVIVAL and PROGRESS.
    Provides clear, continuous signals for the GA to learn from.
    """
    # 1. --- TERMINAL STATE CHECK ---
    # The ultimate goal is to win. This is the strongest signal.
    if current_game_state.is_win():
        return 10000
    if current_game_state.is_lose():
        return -10000

    # 2. --- FEATURE EXTRACTION ---
    pacman_pos = current_game_state.get_pacman_position()
    food_list = current_game_state.get_food().as_list()
    ghost_states = current_game_state.get_ghost_states()
    
    active_ghosts = [g for g in ghost_states if g.scared_timer == 0]
    scared_ghosts = [g for g in ghost_states if g.scared_timer > 0]

    # 3. --- CORE MOTIVATION: SCORE ---
    # The game score itself is a powerful feature.
    score_feature = current_game_state.get_score()

    # 4. --- PRIMARY GOAL: SURVIVAL ---
    # Avoid active ghosts. The penalty should be continuous and escalate sharply.
    survival_feature = 0
    if active_ghosts:
        min_ghost_dist = min(manhattan_distance(pacman_pos, g.get_position()) for g in active_ghosts)
        
        # Emergency signal: If a ghost is about to eat Pacman, this is the worst possible state.
        if min_ghost_dist <= 1:
            return -10000 
        
        # The penalty is the inverse of the distance. A ghost 2 squares away is twice as bad as one 4 squares away.
        survival_feature = -1.0 / min_ghost_dist

    # 5. --- PRIMARY GOAL: PROGRESS ---
    # Eat all the food. This is measured by how much food is left and how close the nearest pellet is.
    progress_feature = 0
    if food_list:
        # Fewer food pellets left is better.
        food_left_penalty = len(food_list)
        # Being closer to the nearest food pellet is better.
        dist_to_food_incentive = min(manhattan_distance(pacman_pos, food) for food in food_list)
        
        progress_feature = -food_left_penalty - dist_to_food_incentive

    # 6. --- BONUS GOAL: HUNT SCARED GHOSTS ---
    # Eating ghosts gives huge points (200). We should incentivize moving towards them when they are scared.
    scared_ghost_feature = 0
    if scared_ghosts:
        # Find closest scared ghost
        min_scared_dist = min(manhattan_distance(pacman_pos, g.get_position()) for g in scared_ghosts)
        # Incentive is inverse distance (closer is better)
        if min_scared_dist > 0:
             scared_ghost_feature = 1.0 / min_scared_dist
        else:
             scared_ghost_feature = 10.0 # On top of the ghost!

    # --- FINAL WEIGHTED EVALUATION ---
    evaluation = (
        weights['score'] * score_feature +
        weights['survival'] * survival_feature +
        weights['progress'] * progress_feature +
        weights.get('scared_ghost', 0) * scared_ghost_feature # Use .get for backward compatibility
    )

    return evaluation


class GeneticAgent(GreedyAgent):
    """
    A Pacman agent that uses weights evolved from a genetic algorithm.
    """
    def __init__(self, layout_name='medium_classic', num_generations=10):
        # --- THIS IS THE FIX ---
        # Call the parent class's (GreedyAgent) constructor.
        # This is essential to set up attributes like 'self.last_action'.
        super().__init__()
        # --- END OF FIX ---

        print("Initializing GeneticAgent...")
        print("This may take a few minutes as it runs the genetic algorithm.")
        
        start_time = time.time()
        
        # Run the GA to find the best weights. This is the core of the agent's "training".
        self.best_weights = genetic_algorithm.run_genetic_algorithm(
            num_generations=int(num_generations),
            layout_name=layout_name
        )
        
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        print("Now playing the game with the evolved weights...")
        
        # Set the evaluation function for the GreedyAgent to use our evolved weights
        self.evaluation_function = lambda state: genetic_evaluation_function(state, self.best_weights)


class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def get_action(self, state):
        legal = state.get_legal_pacman_actions()
        current = state.get_pacman_state().configuration.direction
        if current == Directions.STOP:
            current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal:
            return left
        if current in legal:
            return current
        if Directions.RIGHT[current] in legal:
            return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal:
            return Directions.LEFT[left]
        return Directions.STOP


class GreedyAgent(Agent):
    def __init__(self, eval_fn="score_evaluation"):
        if isinstance(eval_fn, str):
            self.evaluation_function = util.lookup(eval_fn, globals())
        else:
            self.evaluation_function = eval_fn
        assert self.evaluation_function is not None
        # --- NEW: Add memory for the last action ---
        self.last_action = None
        # --- NEW: Add memory for visited positions to prevent oscillation ---
        self.visited_history = [] 

    def get_action(self, state):
        # Generate candidate actions
        legal = state.get_legal_pacman_actions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        successors = [(state.generate_successor(0, action), action) for action in legal]
        
        # Evaluate successors
        scored = []
        for next_state, action in successors:
            score = self.evaluation_function(next_state)
            
            # --- OSCILLATION PREVENTION ---
            # Penalize revisiting recent positions
            pacman_pos = next_state.get_pacman_position()
            if pacman_pos in self.visited_history:
                # Apply a penalty based on how recently it was visited
                # More recent = higher penalty
                recency = self.visited_history.index(pacman_pos)
                penalty = (len(self.visited_history) - recency) * 10 
                score -= penalty
            
            scored.append((score, action))

        # Find the best score and all actions that lead to it
        best_score = max(scored)[0]
        best_actions = [pair[1] for pair in scored if pair[0] == best_score]

        # --- THIS IS THE FIX FOR OSCILLATION ---
        # If there's more than one "best" action, and we have a last action,
        # try to remove the action that would reverse our last move.
        if len(best_actions) > 1 and self.last_action is not None:
            reverse_action = Directions.REVERSE[self.last_action]
            if reverse_action in best_actions:
                best_actions.remove(reverse_action)
        # --- END OF FIX ---

        # Choose one of the remaining best actions randomly
        chosen_action = random.choice(best_actions)
        
        # Remember the chosen action for the next turn
        self.last_action = chosen_action
        
        # Update visited history
        current_pos = state.get_pacman_position()
        self.visited_history.append(current_pos)
        if len(self.visited_history) > 10: # Keep last 10 positions
            self.visited_history.pop(0)
        
        return chosen_action

def score_evaluation(state):
    return state.get_score()


import time
import genetic_algorithm  # Import the new GA file
from util import manhattan_distance


def genetic_evaluation_function(current_game_state, weights):
    """
    --- FINAL, SIMPLIFIED EVALUATION FUNCTION ---
    Focuses on the two most important goals: SURVIVAL and PROGRESS.
    Provides clear, continuous signals for the GA to learn from.
    """
    # 1. --- TERMINAL STATE CHECK ---
    # The ultimate goal is to win. This is the strongest signal.
    if current_game_state.is_win():
        return 10000
    if current_game_state.is_lose():
        return -10000

    # 2. --- FEATURE EXTRACTION ---
    pacman_pos = current_game_state.get_pacman_position()
    food_list = current_game_state.get_food().as_list()
    ghost_states = current_game_state.get_ghost_states()
    
    active_ghosts = [g for g in ghost_states if g.scared_timer == 0]
    scared_ghosts = [g for g in ghost_states if g.scared_timer > 0]

    # 3. --- CORE MOTIVATION: SCORE ---
    # The game score itself is a powerful feature.
    score_feature = current_game_state.get_score()

    # 4. --- PRIMARY GOAL: SURVIVAL ---
    # Avoid active ghosts. The penalty should be continuous and escalate sharply.
    survival_feature = 0
    if active_ghosts:
        min_ghost_dist = min(manhattan_distance(pacman_pos, g.get_position()) for g in active_ghosts)
        
        # Emergency signal: If a ghost is about to eat Pacman, this is the worst possible state.
        if min_ghost_dist <= 1:
            return -10000 
        
        # The penalty is the inverse of the distance. A ghost 2 squares away is twice as bad as one 4 squares away.
        survival_feature = -1.0 / min_ghost_dist

    # 5. --- PRIMARY GOAL: PROGRESS ---
    # Eat all the food. This is measured by how much food is left and how close the nearest pellet is.
    progress_feature = 0
    if food_list:
        # Fewer food pellets left is better.
        food_left_penalty = len(food_list)
        # Being closer to the nearest food pellet is better.
        dist_to_food_incentive = min(manhattan_distance(pacman_pos, food) for food in food_list)
        
        progress_feature = -food_left_penalty - dist_to_food_incentive

    # 6. --- BONUS GOAL: HUNT SCARED GHOSTS ---
    # Eating ghosts gives huge points (200). We should incentivize moving towards them when they are scared.
    scared_ghost_feature = 0
    if scared_ghosts:
        # Find closest scared ghost
        min_scared_dist = min(manhattan_distance(pacman_pos, g.get_position()) for g in scared_ghosts)
        # Incentive is inverse distance (closer is better)
        if min_scared_dist > 0:
             scared_ghost_feature = 1.0 / min_scared_dist
        else:
             scared_ghost_feature = 10.0 # On top of the ghost!

    # --- FINAL WEIGHTED EVALUATION ---
    evaluation = (
        weights['score'] * score_feature +
        weights['survival'] * survival_feature +
        weights['progress'] * progress_feature +
        weights.get('scared_ghost', 0) * scared_ghost_feature # Use .get for backward compatibility
    )

    return evaluation


class GeneticAgent(GreedyAgent):
    """
    A Pacman agent that uses weights evolved from a genetic algorithm.
    """
    def __init__(self, layout_name='medium_classic', num_generations=10):
        # --- THIS IS THE FIX ---
        # Call the parent class's (GreedyAgent) constructor.
        # This is essential to set up attributes like 'self.last_action'.
        super().__init__()
        # --- END OF FIX ---

        print("Initializing GeneticAgent...")
        print("This may take a few minutes as it runs the genetic algorithm.")
        
        start_time = time.time()
        
        # Run the GA to find the best weights. This is the core of the agent's "training".
        self.best_weights = genetic_algorithm.run_genetic_algorithm(
            num_generations=int(num_generations),
            layout_name=layout_name
        )
        
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        print("Now playing the game with the evolved weights...")
        
        # Set the evaluation function for the GreedyAgent to use our evolved weights
        self.evaluation_function = lambda state: genetic_evaluation_function(state, self.best_weights)


class TrainedAgent(GreedyAgent):

    def __init__(self):
        super().__init__()
        best_weights = {
            'score': 1.3257363700298352, 
            'survival': 2.146035445037556, 
            'progress': 1.155112286978574,
            'scared_ghost': 3.9568939887763346
        }
        # ------------------------------------
        
        # Set the evaluation function to use our evolved weights
        self.evaluation_function = lambda state: genetic_evaluation_function(state, best_weights)