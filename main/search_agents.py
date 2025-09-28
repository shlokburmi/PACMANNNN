# search_agents.py
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


from __future__ import print_function
from builtins import str
from future.utils import raise_
from builtins import object
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def get_action(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.get_legal_pacman_actions():
            return Directions.WEST
        else:
            return Directions.STOP

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.
    As a default, this agent runs depth first search on a PositionSearchProblem to
    find location (1,1)
    Options for fn include:
      depth_first_search or dfs
      breadth_first_search or bfs
    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depth_first_search', prob='PositionSearchProblem', heuristic='null_heuristic'):
        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.search_function = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in search_agents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.search_function = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.search_type = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def register_initial_state(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!
        state: a GameState object (pacman.py)
        """
        if self.search_function == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.search_type(state) # Makes a new search problem
        self.actions  = self.search_function(problem) # Find a path
        total_cost = problem.get_cost_of_actions(self.actions)
        print('Path found with total cost of %d in %.2f seconds' % (total_cost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def get_action(self, state):
        """
        Returns the next action in the path chosen earlier (in
        register_initial_state).  Return Directions.STOP if there is no further
        action to take.
        state: a GameState object (pacman.py)
        """
        if 'action_index' not in dir(self): self.action_index = 0
        i = self.action_index
        self.action_index += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.
    The state space consists of (x,y) positions in a pacman game.
    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, game_state, cost_fn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.
        game_state: A GameState object (pacman.py)
        cost_fn: A function from a search state (tuple) to a non-negative number
        goal: A position in the game_state
        """
        self.walls = game_state.get_walls()
        self.start_state = game_state.get_pacman_position()
        if start != None: self.start_state = start
        self.goal = goal
        self.cost_fn = cost_fn
        self.visualize = visualize
        if warn and (game_state.get_num_food() != 1 or not game_state.has_food(goal[0], goal[1])):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def get_start_state(self):
        return self.start_state

    def is_goal_state(self, state):
        is_goal = state == self.goal

        # For display purposes only
        if is_goal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'draw_expanded_cells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.draw_expanded_cells(self._visitedlist) #@UndefinedVariable

        return is_goal

    def get_successors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.
         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.direction_to_vector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                next_state = (nextx, nexty)
                cost = self.cost_fn(next_state)
                successors.append( ( next_state, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def get_cost_of_actions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.get_start_state()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.cost_fn((x,y))
        return cost

def maze_distance(point1, point2, game_state):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The game_state can be any game state -- Pacman's
    position in that state is ignored.
    Example usage: maze_distance( (2,4), (5,6), game_state)
    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = game_state.get_walls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(game_state, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.breadth_first_search(prob))

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.
    A search state in this problem is a tuple ( pacman_position, food_grid ) where
      pacman_position: a tuple (x,y) of integers specifying Pacman's position
      food_grid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, starting_game_state):
        self.start = (starting_game_state.get_pacman_position(), starting_game_state.get_food())
        self.walls = starting_game_state.get_walls()
        self.starting_game_state = starting_game_state
        self._expanded = 0 # DO NOT CHANGE
        self.heuristic_info = {} # A dictionary for the heuristic to store information

    def get_start_state(self):
        return self.start

    def is_goal_state(self, state):
        return state[1].count() == 0

    def get_successors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.direction_to_vector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                next_food = state[1].copy()
                next_food[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), next_food), direction, 1) )
        return successors

    def get_cost_of_actions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.get_start_state()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your food_heuristic"
    def __init__(self):
        self.search_function = lambda prob: search.a_star_search(prob, food_heuristic)
        self.search_type = FoodSearchProblem

class SMHAFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using SMHA* and your food_heuristic"
    def __init__(self):
        self.search_function = lambda prob: search.smha_search(prob, heuristics=[food_heuristic, mst_food_heuristic])
        self.search_type = FoodSearchProblem

def food_heuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.
    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.
    If using A* search, the heuristic values must be non-negative and admissible
    (i.e., less than or equal to the true cost to reach the goal).
    """
    position, food_grid = state
    food_list = food_grid.as_list()
    if not food_list:
        return 0
    
    # Heuristic: distance to the furthest food dot
    max_dist = 0
    for food in food_list:
        dist = util.manhattanDistance(position, food)
        if dist > max_dist:
            max_dist = dist
    return max_dist

def mst_food_heuristic(state, problem):
    "A more advanced heuristic for the FoodSearchProblem that uses the Minimum Spanning Tree (MST)"
    position, food_grid = state
    food_list = food_grid.as_list()
    if not food_list:
        return 0

    nodes = [position] + food_list
    if len(nodes) == 2:
        return maze_distance(nodes[0], nodes[1], problem.starting_game_state)

    if 'distances' not in problem.heuristic_info:
        problem.heuristic_info['distances'] = {}
    
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            pos1, pos2 = nodes[i], nodes[j]
            if (pos1, pos2) not in problem.heuristic_info['distances']:
                dist = maze_distance(pos1, pos2, problem.starting_game_state)
                problem.heuristic_info['distances'][(pos1, pos2)] = dist
                problem.heuristic_info['distances'][(pos2, pos1)] = dist
            
            dist = problem.heuristic_info['distances'][(pos1, pos2)]
            edges.append((dist, pos1, pos2))

    mst_cost = 0
    parent = {node: node for node in nodes}

    def find_set(v):
        if v == parent[v]:
            return v
        parent[v] = find_set(parent[v])
        return parent[v]

    def union_sets(a, b):
        a = find_set(a)
        b = find_set(b)
        if a != b:
            parent[b] = a

    edges.sort()

    for cost, u, v in edges:
        if find_set(u) != find_set(v):
            mst_cost += cost
            union_sets(u, v)
            
    min_dist_to_mst = float('inf')
    for food in food_list:
        dist = maze_distance(position, food, problem.starting_game_state)
        if dist < min_dist_to_mst:
            min_dist_to_mst = dist
            
    return mst_cost + min_dist_to_mst

